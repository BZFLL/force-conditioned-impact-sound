#!/usr/bin/env python3
"""Synthetic pretraining corpus for the DDSP backbone.

The recipe is recorded in
experiments/pretrain/pretrain_synthetic/pretrain_config.json. Each sample is
a decaying noise burst convolved with one random room impulse response from
the synthetic IR pool (make_synthetic_ir_pool.py):

  excitation   decaying noise burst: tau log-uniform 3-100 ms, one-pole
               lowpass with coefficient a ~ U[0, 0.95], peak log-uniform
               in [0.05, 0.5] (a 20 dB range), onset at 5 ms
  force input  the burst itself, as the 200 ms force window
  target       the burst convolved with a random pool IR, truncated to the
               200 ms window

The corpus mirrors the held-out split sizes (2936 train / 304 val / 308
test). Sampling is deterministic given (base_seed, epoch, index): the
training portion is redrawn every epoch (train.py calls set_epoch), while
the validation and test portions are fixed so that best-model selection is
stable.

The class mirrors the interface of dataset.ObjectFolderRealDataset that
train.py relies on: __getitem__ dicts (mel, waveform, force_waveform,
force_features, obj_id, material_idx, striking_N), set_mel_stats /
get_raw_mel, n_objects / n_materials.
"""

import numpy as np
import torch
import torchaudio

SR = 48000
N_SAMPLES = 9600                  # 200 ms
ONSET_SAMPLE = int(0.005 * SR)    # onset at 5 ms
N_MELS, N_FFT, HOP = 128, 2048, 256

N_TRAIN, N_VAL, N_TEST = 2936, 304, 308   # held-out split sizes


def sample_burst(rng, sr=SR, n=N_SAMPLES):
    """One decaying-noise-burst excitation (float64, length n)."""
    tau = np.exp(rng.uniform(np.log(3e-3), np.log(100e-3)))   # 3-100 ms
    a = rng.uniform(0.0, 0.95)                                # one-pole LP
    peak = np.exp(rng.uniform(np.log(0.05), np.log(0.5)))     # 20 dB range
    t = np.arange(n) / sr
    burst = rng.standard_normal(n) * np.exp(-t / tau)
    # one-pole lowpass: y[n] = (1-a) x[n] + a y[n-1]
    from scipy.signal import lfilter
    burst = lfilter([1.0 - a], [1.0, -a], burst)
    out = np.zeros(n)
    body = burst[: n - ONSET_SAMPLE]
    m = np.abs(body).max()
    if m > 0:
        body = body / m * peak
    out[ONSET_SAMPLE:ONSET_SAMPLE + len(body)] = body
    return out


class SyntheticImpactDataset(torch.utils.data.Dataset):
    """On-the-fly synthetic corpus (see module docstring)."""

    def __init__(self, ir_pool, base_seed=4243,
                 n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST):
        """ir_pool: path to cache/synthetic_ir_pool.pt or the (N, 9600)
        array itself."""
        if isinstance(ir_pool, (str, bytes)):
            blob = torch.load(ir_pool, map_location="cpu", weights_only=False)
            ir_pool = blob["irs"].numpy()
        self.irs = np.asarray(ir_pool, dtype=np.float64)
        self.base_seed = int(base_seed)
        self.n_train, self.n_val, self.n_test = n_train, n_val, n_test
        self.epoch = 0
        self._cache = {}
        self.mel_mean = None
        self.mel_std = None
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            power=2.0)
        # attributes train.py reads when building the model
        self.n_objects = 100
        self.n_materials = 7

    def __len__(self):
        return self.n_train + self.n_val + self.n_test

    def set_epoch(self, epoch):
        """Redraw the training portion; validation/test stay fixed."""
        self.epoch = int(epoch)
        self._cache = {}

    def set_mel_stats(self, mean, std):
        self.mel_mean, self.mel_std = mean, std

    def _rng_for(self, idx):
        # epoch affects the training indices only
        epoch = self.epoch if idx < self.n_train else 0
        return np.random.default_rng(
            np.random.SeedSequence([self.base_seed, epoch, int(idx)]))

    def _make(self, idx):
        rng = self._rng_for(idx)
        burst = sample_burst(rng)
        ir = self.irs[rng.integers(len(self.irs))]
        wav = np.convolve(burst, ir)[:N_SAMPLES]

        # v1 6-D descriptor of the burst (same extractor as dataset.py)
        from dataset import extract_force_features
        f = extract_force_features(burst, SR)
        force_vec = torch.tensor([f.peak, f.fwhm_ms, float(f.bounce_count),
                                  f.max_bounce_ratio, f.tau_ms, f.asymmetry],
                                 dtype=torch.float32)

        audio = torch.tensor(wav, dtype=torch.float32)
        mel = torch.log1p(self.mel_transform(audio))
        return {
            "force_features": force_vec,
            "force_waveform": torch.tensor(burst, dtype=torch.float32),
            "mel": mel,                                   # log1p, raw
            "waveform": audio,
            "obj_id": int(rng.integers(self.n_objects)),
            "obj_id_raw": int(rng.integers(self.n_objects)),
            "material_idx": int(rng.integers(self.n_materials)),
            "impact_id": str(idx),
            "striking_N": 0.0,
        }

    def _cached(self, idx):
        if idx not in self._cache:
            self._cache[idx] = self._make(idx)
        return self._cache[idx]

    def get_raw_mel(self, idx):
        return self._cached(idx)["mel"]

    def __getitem__(self, idx):
        item = self._cached(idx)
        if self.mel_mean is not None:
            item = dict(item)
            item["mel"] = (item["mel"] - self.mel_mean) / self.mel_std
        return item
