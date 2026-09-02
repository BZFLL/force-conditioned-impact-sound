"""ObjectFolder Real dataset for force-conditioned impact sound generation.

Each sample is a dict with the target log1p mel spectrogram of the 200 ms
onset window ('mel', shape (n_mels, T)), the force descriptor vector
('force_features'), the raw 200 ms force window ('force_waveform'), the
object and material indices ('obj_id', 'material_idx'; 'obj_id_raw' and
'impact_id' identify the recording) and, with return_waveform=True, the raw
48 kHz onset-window audio ('waveform') from which the mel was computed.

The mic onset window and the force window share one anchor, the force peak:
window start = force peak minus 5 ms. The default 200 ms force window at 32
samples per frame gives 300 force frames, the synthesis frame rate.

Force feature sets (feature_set argument):
  v1      6-D descriptor [peak, FWHM, bounce_count, max_bounce_ratio, tau,
          asymmetry] computed on the full force recording (default).
  impulse 1-D: integral of |F| over the 200 ms window.
  v2A     3-D: peak, contact duration (width at 10% of peak), rise/fall
          asymmetry, computed on the 200 ms window.
  v2AB    6-D: v2A plus impulse, bounce_count, max_bounce_ratio.
  pca6    6-D: scores of the pooled force curve (per-frame RMS and max|F|)
          on a train-split PCA basis (fit by fit_pca_basis.py).

Normalisation: mels are log1p only; a global per-mel-bin z-score with
train-split statistics (compute_mel_stats) is applied in __getitem__ once
set_mel_stats() has been called. Force descriptors, force windows and target
waveforms are returned raw; train.py / evaluate.py z-score or scale them with
train-split statistics. No per-sample peak normalisation anywhere. Splits are
read from splits/splits_<mode>.json written by make_splits.py
(load_split_indices). All samples are loaded and cached in memory at
construction.
"""

import os
import glob
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset
from scipy import signal as sig
from dataclasses import dataclass

# Dataset location: repository-relative by default, overridable via env vars.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = os.environ.get("OFR_ROOT", str(PROJECT_ROOT / "data" / "ObjectFolder_Real"))
# objects.csv (no header): col0=obj_id, col3=material. Used for the material
# embedding and material-stratified splits. 7 materials over the 100
# ObjectFolder_Real objects.
OBJECTS_CSV = os.environ.get("OFR_OBJECTS_CSV", str(PROJECT_ROOT / "data" / "ObjectFolder" / "objects.csv"))
MATERIALS = ["Ceramic", "Glass", "Iron", "Plastic", "Polycarbonate",
             "Steel", "Wood"]  # sorted; fixed index order (7 classes)
MATERIAL_TO_IDX = {m: i for i, m in enumerate(MATERIALS)}


@dataclass
class ForceFeatures:
    """Multi-scale force features extracted from Force.wav."""
    peak: float           # Peak amplitude (proportional to Newtons)
    fwhm_ms: float        # Full-width at half-maximum (ms)
    bounce_count: int     # Number of secondary bounces
    max_bounce_ratio: float  # Largest bounce relative to peak
    tau_ms: float         # Exponential decay time constant (ms)
    asymmetry: float      # Rise rate / fall rate
    striking_N: float     # Metadata striking force (Newtons)


def extract_force_features(force_wav: np.ndarray, sr: int) -> ForceFeatures:
    """Extract multi-scale features from a force waveform."""
    abs_data = np.abs(force_wav)
    peak_idx = abs_data.argmax()
    peak_val = abs_data[peak_idx]

    if peak_val < 1e-6:
        return ForceFeatures(0, 0, 0, 0, 0, 1.0, 0)

    # FWHM: walk outward from the main peak until the envelope drops below
    # half-max (contiguous span). A first-to-last-crossing span would stretch
    # FWHM to the peak-to-bounce distance when a secondary bounce exceeds
    # half-max.
    half_max = peak_val / 2
    l = peak_idx
    while l > 0 and abs_data[l - 1] > half_max:
        l -= 1
    r = peak_idx
    while r < len(abs_data) - 1 and abs_data[r + 1] > half_max:
        r += 1
    fwhm_ms = (r - l + 1) / sr * 1000

    # Secondary bounces
    search_start = peak_idx + int(0.002 * sr)
    search_end = min(len(force_wav), peak_idx + int(0.2 * sr))
    if search_start < search_end:
        segment = abs_data[search_start:search_end]
        local_peaks, _ = sig.find_peaks(
            segment,
            height=peak_val * 0.05,
            distance=int(0.001 * sr),
            prominence=peak_val * 0.02
        )
        bounce_count = len(local_peaks)
        max_bounce_ratio = segment[local_peaks].max() / peak_val if bounce_count > 0 else 0
    else:
        bounce_count = 0
        max_bounce_ratio = 0

    # Decay time constant
    decay_start = peak_idx + 1
    decay_end = min(len(force_wav), peak_idx + int(0.05 * sr))
    tau_ms = 30.0  # default
    if decay_end > decay_start + 10:
        t_decay = np.arange(decay_end - decay_start) / sr
        f_decay = abs_data[decay_start:decay_end]
        valid = f_decay > peak_val * 0.001
        if valid.sum() > 10:
            try:
                log_f = np.log(f_decay[valid])
                coeffs = np.polyfit(t_decay[valid], log_f, 1)
                decay_rate = -coeffs[0]
                if decay_rate > 0:
                    tau_ms = 1000 / decay_rate
            except Exception:
                pass

    # Rise/fall asymmetry (dF/dt)
    pre = max(0, peak_idx - int(0.005 * sr))
    post = min(len(force_wav), peak_idx + int(0.02 * sr))
    pulse = force_wav[pre:post]
    dfdt = np.diff(pulse) * sr
    max_rise = dfdt.max() if len(dfdt) > 0 else 1
    max_fall = abs(dfdt.min()) if len(dfdt) > 0 else 1
    asymmetry = max_rise / max_fall if max_fall > 0 else 1.0

    return ForceFeatures(
        peak=float(peak_val),
        fwhm_ms=float(fwhm_ms),
        bounce_count=int(bounce_count),
        max_bounce_ratio=float(max_bounce_ratio),
        tau_ms=float(min(tau_ms, 100)),  # Cap at 100ms
        asymmetry=float(asymmetry),
        striking_N=0.0  # set by the caller from striking_force.yaml
    )


# Window-matched force descriptors (feature sets impulse / v2A / v2AB),
# computed on the same 200 ms force window the time-resolved (L4) input sees.
# extract_force_features implements the "v1" feature set used by the main grid.
def extract_force_features_v2(force_window: np.ndarray, sr: int) -> dict:
    """Force-pulse descriptors of the 200 ms window (dict). Keys:
      peak             max|F|                       (Stoelinga & Lutfi 2011: Fmax ~ v)
      contact_dur_ms   contiguous width > 10% peak  (Stoelinga & Lutfi 2011: contact
                                                     time sets the low-pass cutoff)
      asymmetry        max rise-rate / max fall-rate (Hunt & Crossley 1975 contact model)
      impulse          integral of |F| dt           (impulse-momentum)
      rise_ms          10% to 90% rise time         (attack-time timbre axis)
      bounce_count     number of secondary peaks    (Warren & Verbrugge 1984)
      max_bounce_ratio largest secondary / peak     (Warren & Verbrugge 1984)
    """
    a = np.abs(force_window)
    pk_i = int(a.argmax()); pk = float(a[pk_i])
    if pk < 1e-9:
        return dict(peak=0.0, contact_dur_ms=0.0, asymmetry=1.0, impulse=0.0,
                    rise_ms=0.0, bounce_count=0.0, max_bounce_ratio=0.0)
    dt = 1.0 / sr
    impulse = float(a.sum() * dt)
    # contact duration at 10% of peak (used instead of FWHM, whose half-max
    # width is quantisation-floored on this data)
    thr = 0.1 * pk
    l = pk_i
    while l > 0 and a[l - 1] > thr:
        l -= 1
    r = pk_i
    while r < len(a) - 1 and a[r + 1] > thr:
        r += 1
    contact_dur_ms = (r - l + 1) / sr * 1000
    # 10-90% rise time on the rising edge
    lo, hi = 0.1 * pk, 0.9 * pk
    j = pk_i
    while j > 0 and a[j] > lo:
        j -= 1
    rise_ms = (pk_i - j) / sr * 1000 if j < pk_i else 0.0
    # rise/fall asymmetry (dF/dt), same definition as v1's asymmetry
    pre = max(0, pk_i - int(0.005 * sr))
    post = min(len(force_window), pk_i + int(0.02 * sr))
    dfdt = np.diff(force_window[pre:post]) * sr
    max_rise = dfdt.max() if len(dfdt) > 0 else 1.0
    max_fall = abs(dfdt.min()) if len(dfdt) > 0 else 1.0
    asymmetry = float(max_rise / max_fall) if max_fall > 0 else 1.0
    # secondary bounces within the 200 ms window (same rule as v1)
    search_start = pk_i + int(0.002 * sr)
    if search_start < len(force_window):
        seg = a[search_start:]
        peaks, _ = sig.find_peaks(seg, height=pk * 0.05,
                                  distance=int(0.001 * sr), prominence=pk * 0.02)
        bounce_count = float(len(peaks))
        max_bounce_ratio = float(seg[peaks].max() / pk) if len(peaks) else 0.0
    else:
        bounce_count = 0.0; max_bounce_ratio = 0.0
    return dict(peak=pk, contact_dur_ms=contact_dur_ms, asymmetry=asymmetry,
                impulse=impulse, rise_ms=rise_ms, bounce_count=bounce_count,
                max_bounce_ratio=max_bounce_ratio)


def pool_force_curve_np(force_window: np.ndarray, w: int = 32) -> np.ndarray:
    """Flattened (n_frames*2,) [RMS, max|F|] per w-sample frame. Must match
    model_ddsp.pool_force_curve. The PCA-6 basis is fit in this space."""
    fr = force_window.reshape(-1, w)
    rms = np.sqrt((fr.astype(np.float64) ** 2).mean(-1))
    mx = np.abs(fr).max(-1)
    return np.stack([rms, mx], -1).reshape(-1)


# feature_set -> ordered dim list (dim 0 = the scalar for that condition).
# "v1" = original 6-dim (handled by extract_force_features, not this table).
# "pca6" = 6-dim linear summary: scores on the train-split PCA basis over
#          pool_force_curve_np space (basis file passed via the dataset's
#          pca_basis arg; fit by fit_pca_basis.py on the train split only).
FEATURE_SETS_V2 = {
    "impulse": ["impulse"],                                        # 1-D
    "v2A":     ["peak", "contact_dur_ms", "asymmetry"],           # 3-D
    # rise_ms is not used: at 48 kHz it quantises to the sample grid and is
    # degenerate on this data.
    "v2AB":    ["peak", "contact_dur_ms", "asymmetry",
                "impulse", "bounce_count", "max_bounce_ratio"],  # 6-D
}


def load_material_map(csv_path: str = OBJECTS_CSV) -> dict:
    """obj_id (int) → material_idx (int, 0..6). Reads objects.csv col0/col3."""
    df = pd.read_csv(csv_path, header=None)
    return {int(r[0]): MATERIAL_TO_IDX[r[3]] for _, r in df.iterrows()
            if r[3] in MATERIAL_TO_IDX}


class ObjectFolderRealDataset(Dataset):
    """
    ObjectFolder Real impacts, all cached in memory at construction.

    __getitem__ returns the log1p mel z-scored with the global train-split
    per-mel-bin statistics set via set_mel_stats(). Before set_mel_stats()
    is called the raw log1p mel is returned; that is what compute_mel_stats()
    consumes.
    """

    def __init__(
        self,
        data_root: str = DATA_ROOT,
        sr: int = 48000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 256,
        onset_duration_ms: float = 200.0,
        force_window_ms: float = 200.0,
        object_ids: list = None,
        return_waveform: bool = False,
        feature_set: str = "v1",
        pca_basis: str = None,
    ):
        self.data_root = data_root
        self.return_waveform = return_waveform
        assert feature_set in ("v1", "pca6") or feature_set in FEATURE_SETS_V2, \
            f"unknown feature_set {feature_set!r}"
        self.feature_set = feature_set
        self._pca_mean = self._pca_comp = None
        if feature_set == "pca6":
            assert pca_basis and os.path.exists(pca_basis), \
                f"feature_set=pca6 needs an existing pca_basis file, got {pca_basis!r}"
            b = torch.load(pca_basis, map_location="cpu", weights_only=False)
            self._pca_mean = b["mean"].numpy().astype(np.float64)      # (600,)
            self._pca_comp = b["components"].numpy().astype(np.float64)  # (6,600)
        # obj_id -> material_idx (0..6). Always populated; models pick
        # which identity (obj vs material) to condition on.
        self.material_map = load_material_map()
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.onset_samples = int(onset_duration_ms / 1000 * sr)
        self.force_window_samples = int(force_window_ms / 1000 * sr)

        # Global mel z-score stats (set later via set_mel_stats)
        self.mel_mean = None
        self.mel_std = None

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        # Find all impact directories
        self.samples = []
        obj_dirs = sorted(glob.glob(os.path.join(data_root, "[0-9]*")))

        for obj_dir in obj_dirs:
            obj_id = int(os.path.basename(obj_dir))
            if object_ids is not None and obj_id not in object_ids:
                continue

            for impact_dir in sorted(glob.glob(os.path.join(obj_dir, "audio", "*"))):
                # Skip non-numeric impact dirs: the dataset ships one
                # deprecated take (24/audio/13_deprecated, a re-recorded
                # impact whose replacement "13" also exists).
                if not os.path.basename(impact_dir).isdigit():
                    continue
                force_path = os.path.join(impact_dir, "Force.wav")
                mic_path = os.path.join(impact_dir, "mic.wav")
                sf_path = os.path.join(impact_dir, "striking_force.yaml")

                if not os.path.exists(force_path) or not os.path.exists(mic_path):
                    continue

                # Load metadata
                striking_N = 0.0
                if os.path.exists(sf_path):
                    with open(sf_path) as f:
                        meta = yaml.safe_load(f)
                    if isinstance(meta, list) and len(meta) > 0:
                        striking_N = meta[0].get('striking_force_newtons', 0.0)

                self.samples.append({
                    'force_path': force_path,
                    'mic_path': mic_path,
                    'obj_id': obj_id,
                    'impact_id': os.path.basename(impact_dir),
                    'impact_dir': impact_dir,
                    'striking_N': striking_N,
                })

        # Build object_id → index mapping
        unique_objs = sorted(set(s['obj_id'] for s in self.samples))
        self.obj_to_idx = {obj: idx for idx, obj in enumerate(unique_objs)}
        self.n_objects = len(unique_objs)
        self.n_materials = len(MATERIALS)  # fixed at 7

        print(f"ObjectFolderRealDataset: {len(self.samples)} samples, "
              f"{self.n_objects} objects, {self.n_materials} materials")

        # Pre-cache all samples into memory for fast GPU training
        print("  Pre-caching all samples to memory...")
        self._cache = [None] * len(self.samples)
        for i in range(len(self.samples)):
            self._cache[i] = self._load_sample(i)
            if (i + 1) % 200 == 0:
                print(f"    cached {i+1}/{len(self.samples)}")
        print(f"  Done! All {len(self.samples)} samples cached.")

    def __len__(self):
        return len(self.samples)

    def set_mel_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set global per-mel-bin z-score stats (each shape (n_mels, 1))."""
        self.mel_mean = mean
        self.mel_std = std

    def get_raw_mel(self, idx):
        """Un-normalized log1p mel (used by compute_mel_stats)."""
        return self._cache[idx]['mel']

    def _get_onset_window(self, data, sr, anchor_idx):
        """Extract the onset window anchored to a given event index.

        The anchor is the force peak, passed in rather than taken from
        `data`'s own argmax: a mic window anchored on its own argmax can
        drift off the force peak or latch onto a late resonance.
        """
        # Start a bit before the (force) peak
        pre_samples = int(0.005 * sr)  # 5ms before peak
        start = max(0, anchor_idx - pre_samples)
        end = start + self.onset_samples

        if end > len(data):
            end = len(data)
            start = max(0, end - self.onset_samples)

        return data[start:end], start

    def __getitem__(self, idx):
        item = self._cache[idx]
        if self.mel_mean is not None:
            # Global z-score with train-split statistics.
            item = dict(item)
            item['mel'] = (item['mel'] - self.mel_mean) / self.mel_std
        return item

    def _load_sample(self, idx):
        sample = self.samples[idx]

        # Load audio
        force_wav, sr_f = sf.read(sample['force_path'])
        mic_wav, sr_m = sf.read(sample['mic_path'])

        assert sr_f == self.sr and sr_m == self.sr, \
            f"Sample rate mismatch: force={sr_f}, mic={sr_m}, expected={self.sr}"

        # Extract force features
        features = extract_force_features(force_wav, self.sr)
        features.striking_N = sample['striking_N']

        # Force feature vector (6 dims)
        force_vec = torch.tensor([
            features.peak,
            features.fwhm_ms,
            float(features.bounce_count),
            features.max_bounce_ratio,
            features.tau_ms,
            features.asymmetry,
        ], dtype=torch.float32)

        # Compute the force peak once and anchor both the force window and
        # the mic onset window to it (force_peak - 5 ms).
        abs_force = np.abs(force_wav)
        force_peak_idx = int(abs_force.argmax())

        # Raw force waveform around the impact (L4 input). The caller
        # z-scores it with train-split statistics.
        fw_start = max(0, force_peak_idx - int(0.005 * self.sr))  # 5ms before
        fw_end = fw_start + self.force_window_samples
        if fw_end > len(force_wav):
            fw_end = len(force_wav)
            fw_start = max(0, fw_end - self.force_window_samples)
        force_window = force_wav[fw_start:fw_end]
        # Pad if needed
        if len(force_window) < self.force_window_samples:
            force_window = np.pad(force_window, (0, self.force_window_samples - len(force_window)))
        force_waveform = torch.tensor(force_window, dtype=torch.float32)

        # The v2 and pca6 feature sets replace the v1 force_vec with one
        # computed on the same 200 ms window L4 sees.
        if self.feature_set == "pca6":
            curve = pool_force_curve_np(force_window, 32)
            force_vec = torch.tensor(
                (curve - self._pca_mean) @ self._pca_comp.T, dtype=torch.float32)
        elif self.feature_set != "v1":
            f2 = extract_force_features_v2(force_window, self.sr)
            force_vec = torch.tensor(
                [f2[k] for k in FEATURE_SETS_V2[self.feature_set]],
                dtype=torch.float32)

        # Onset mel from the mic, anchored to the force peak like the force
        # window above.
        onset_audio, _ = self._get_onset_window(mic_wav, self.sr, force_peak_idx)

        # Pad to exact onset length
        if len(onset_audio) < self.onset_samples:
            onset_audio = np.pad(onset_audio, (0, self.onset_samples - len(onset_audio)))

        # To mel spectrogram
        audio_tensor = torch.tensor(onset_audio, dtype=torch.float32)
        mel = self.mel_transform(audio_tensor)

        # Log-scale mel; the global z-score is applied in __getitem__.
        mel = torch.log1p(mel)

        # Object ID
        obj_idx = self.obj_to_idx[sample['obj_id']]
        # material index (0..6) for the material embedding
        material_idx = self.material_map[sample['obj_id']]

        result = {
            'force_features': force_vec,        # (6,)
            'force_waveform': force_waveform,   # (force_window_samples,)
            'mel': mel,                          # (n_mels, T), log1p, un-normalised
            'obj_id': obj_idx,                   # int
            'obj_id_raw': sample['obj_id'],      # original int
            'material_idx': material_idx,        # int, 0..6
            'impact_id': sample['impact_id'],    # str
            'striking_N': features.striking_N,
        }

        if self.return_waveform:
            # DDSP backbone target: the same onset-window tensor the mel
            # above was computed from. train.py applies the global
            # train-split std scaling.
            result['waveform'] = audio_tensor   # (onset_samples,) @ 48 kHz

        return result


def compute_mel_stats(dataset, train_indices):
    """
    Global per-mel-bin mean/std of log1p mel over the train split only.

    Returns (mean, std), each of shape (n_mels, 1) so they broadcast over
    time frames. Must be computed before set_mel_stats() is called.
    """
    mels = torch.stack([dataset.get_raw_mel(i) for i in train_indices])  # (N, n_mels, T)
    mean = mels.mean(dim=(0, 2)).unsqueeze(1)  # (n_mels, 1)
    std = mels.std(dim=(0, 2)).unsqueeze(1)    # (n_mels, 1)
    std = std.clamp(min=1e-6)
    print(f"compute_mel_stats: {len(train_indices)} train samples, "
          f"mean range [{mean.min():.4f}, {mean.max():.4f}], "
          f"std range [{std.min():.4f}, {std.max():.4f}]")
    return mean, std


def load_split_indices(dataset, split_json_path):
    """
    Map a splits_*.json file (generated by make_splits.py) to dataset
    sample indices. Returns (train_idx, val_idx, test_idx).

    All ablation levels use the same split file.
    """
    with open(split_json_path) as f:
        spec = json.load(f)

    key_to_idx = {(s['obj_id'], s['impact_id']): i
                  for i, s in enumerate(dataset.samples)}

    out = {}
    for split in ('train', 'val', 'test'):
        idx = []
        for obj_str, impact_ids in spec['splits'][split].items():
            obj_id = int(obj_str)
            for imp in impact_ids:
                key = (obj_id, imp)
                if key not in key_to_idx:
                    raise KeyError(
                        f"Split entry not found in dataset: obj={obj_id}, "
                        f"impact={imp} (regenerate splits with make_splits.py?)")
                idx.append(key_to_idx[key])
        out[split] = sorted(idx)

    print(f"Split '{spec['mode']}' (seed={spec['seed']}) loaded from "
          f"{os.path.basename(split_json_path)}: "
          f"train={len(out['train'])}, val={len(out['val'])}, test={len(out['test'])}")
    return out['train'], out['val'], out['test']


if __name__ == "__main__":
    # Quick test
    ds = ObjectFolderRealDataset()
    print(f"\nTotal samples: {len(ds)}")
    sample = ds[0]
    print(f"Force features shape: {sample['force_features'].shape}")
    print(f"Force features: {sample['force_features']}")
    print(f"Force waveform shape: {sample['force_waveform'].shape}")
    print(f"Mel shape: {sample['mel'].shape} (un-normalized: "
          f"min={sample['mel'].min():.3f}, max={sample['mel'].max():.3f})")
    print(f"Object ID: {sample['obj_id']} (raw: {sample['obj_id_raw']}, "
          f"impact: {sample['impact_id']})")
    print(f"Striking force: {sample['striking_N']} N")

    split_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "splits", "splits_heldout.json")
    if os.path.exists(split_file):
        train_idx, val_idx, test_idx = load_split_indices(ds, split_file)
        mean, std = compute_mel_stats(ds, train_idx)
        ds.set_mel_stats(mean, std)
        m = ds[0]['mel']
        print(f"z-scored mel: mean={m.mean():.3f}, std={m.std():.3f}")
    else:
        print(f"(run make_splits.py first to test split loading: {split_file})")
