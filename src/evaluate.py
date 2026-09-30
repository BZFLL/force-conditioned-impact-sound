"""Evaluation of trained checkpoints on the test or validation split.

Computes object-level Fréchet Audio Distance (VGGish embeddings; per-object
FAD averaged over objects), paired bootstrap over objects with Holm
correction for the L2/L3/L4-vs-L1 family, Cohen's d and Cliff's delta, plus
secondary signal metrics (onset error, attack log-mel L1 with a
shift-aligned control, band-energy L1), a per-material breakdown and a
nearest-force retrieval reference. Seeds are treated as a random effect
(per-object FAD averaged over seeds before the bootstrap).

DDSP checkpoints generate waveforms directly (eval mode, loop_offset=0, no
sampling); CVAE checkpoints generate mel spectrograms that are un-z-scored
with the train-split mel statistics, expm1'd and inverted with
InverseMelScale + Griffin-Lim (fixed seed). Both are scored against the same
200 ms force-anchored ground-truth window. CVAE and DDSP are compared on FAD
only; their training losses are not comparable.

The FAD encoder is VGGish (128-d, torch.hub 'harritaylor/torchvggish'); clips
are resampled to 16 kHz with torchaudio beforehand. If VGGish cannot be
loaded the script falls back to a log-mel-statistics encoder and records
fad_encoder="logmel_fallback" / fad_encoder_compliant=false in the outputs;
such FADs are not comparable to VGGish-FAD.

Usage:
    python3 src/evaluate.py --backbone ddsp --split within \
        --ckpt experiments/ablation_1_within_ddsp_s42 experiments/ablation_2_within_ddsp_s42 ... [--dev]
    (scripts/eval_test_main.sh runs the four backbone x split combinations on
    the 80 main-grid checkpoints; --ckpt accepts run directories or a directory
    that contains only the runs to be pooled by their "ablation" level)
"""

import os
import sys
import json
import types
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio

# Local imports (same dir)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (ObjectFolderRealDataset, DATA_ROOT, MATERIALS,
                     compute_mel_stats, load_split_indices)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "splits"
STATS_DIR = PROJECT_ROOT / "stats"
EXP_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"

# Evaluation constants
SR = 48000                 # dataset sample rate
VGGISH_SR = 16000          # VGGish front-end sample rate
ONSET_TOL_MS = 5.0         # onset tolerance ±5 ms
ATTACK_WIN_MS = 10.0       # attack window (first 10 ms log-mel L1)
N_BOOTSTRAP = 2000         # paired bootstrap over objects
PER_MATERIAL_MIN_OBJ = 3   # report per-material CI only if n_obj >= 3
PRIMARY_FAMILY = [2, 3, 4]  # Holm family: L2/L3/L4 vs L1
GL_SEED = 42               # deterministic Griffin-Lim
GL_ITERS = 60              # Griffin-Lim iterations (fixed)

# Mel transform params must match dataset.py exactly (for GL inversion + metrics)
N_FFT = 2048
HOP = 256
N_MELS = 128


# 1. FAD encoder (primary = VGGish; fallback = log-mel stats)

class VGGishEncoder:
    """VGGish (128-d) via torch.hub, on clips pre-resampled to 16 kHz."""

    name = "vggish"

    def __init__(self, device):
        # VGGish runs on CPU so the embeddings are device-independent.
        self.device = torch.device("cpu")
        # Stub resampy before importing torchvggish (it is an import-time
        # dependency that may not be installable; we do our own resampling
        # so the stub is never reached).
        import importlib.util as _ilu
        if _ilu.find_spec("resampy") is None and "resampy" not in sys.modules:
            stub = types.ModuleType("resampy")
            # A real __spec__ so torch.hub's import machinery doesn't choke on
            # `resampy.__spec__ is None`.
            stub.__spec__ = _ilu.spec_from_loader("resampy", loader=None)
            def _blocked(*a, **k):
                raise RuntimeError(
                    "resampy stub reached — VGGish input must be "
                    "pre-resampled to 16 kHz by evaluate.py.")
            stub.resample = _blocked
            sys.modules["resampy"] = stub

        hub_dir = Path(torch.hub.get_dir()) / "harritaylor_torchvggish_master"
        if hub_dir.exists():
            sys.path.insert(0, str(hub_dir))
        # torch.hub.load downloads the repo (+ weights) if not cached, else uses cache.
        self.model = torch.hub.load(
            "harritaylor/torchvggish", "vggish",
            postprocess=False, preprocess=False,
            trust_repo=True, verbose=False)
        # torchvggish's VGGish keeps its own .device attribute (which
        # defaults to CUDA when available) and does x.to(self.device)
        # inside forward(). Pin both the attribute and the weights to CPU
        # so the embeddings are identical regardless of the training device.
        self.model.device = self.device
        self.model.to(self.device).eval()
        from torchvggish import vggish_input  # noqa
        self._vggish_input = vggish_input
        self.resampler = torchaudio.transforms.Resample(SR, VGGISH_SR).to(self.device)

    @torch.no_grad()
    def embed(self, wav_48k: torch.Tensor) -> np.ndarray:
        """wav_48k: (T,) float tensor at 48 kHz → 128-d embedding (np, averaged
        over VGGish's 0.96 s frames; our onset window is 200 ms → 1 frame)."""
        wav = wav_48k.to(self.device).float()
        wav = wav / (wav.abs().max() + 1e-9)          # VGGish expects ~[-1,1]
        wav16 = self.resampler(wav)
        # VGGish forms 0.96 s log-mel examples; our onset window is 200 ms, too
        # short for even one example. Zero-pad the 16 kHz clip to >= 0.975 s so
        # exactly one example is produced (standard short-clip FAD handling).
        min_len = int(0.975 * VGGISH_SR) + 1
        if wav16.numel() < min_len:
            wav16 = torch.cat([wav16, wav16.new_zeros(min_len - wav16.numel())])
        wav16 = wav16.cpu().numpy().astype(np.float32)
        ex = self._vggish_input.waveform_to_examples(
            wav16, VGGISH_SR, return_tensor=True).to(self.device).float()
        emb = self.model(ex)                          # (n_frames, 128)
        return emb.mean(dim=0).cpu().numpy()          # (128,)


class LogMelFallbackEncoder:
    """Fallback only, used if VGGish cannot be loaded; not comparable to
    VGGish-FAD. Concatenated per-mel-bin [mean, std] of the log-mel = a crude
    128*2 = 256-d "embedding". Every output that uses this is flagged
    (fad_encoder_compliant=false)."""

    name = "logmel_fallback"

    def __init__(self, device):
        self.device = device
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            power=2.0).to(device)

    @torch.no_grad()
    def embed(self, wav_48k: torch.Tensor) -> np.ndarray:
        wav = wav_48k.to(self.device).float()
        m = torch.log1p(self.mel(wav))                # (n_mels, T)
        return torch.cat([m.mean(1), m.std(1)]).cpu().numpy()  # (256,)


def build_fad_encoder(device):
    """Try VGGish (primary). On any failure, fall back and flag it."""
    try:
        enc = VGGishEncoder(device)
        print("[FAD] encoder = VGGish (primary encoder)")
        return enc, False
    except Exception as e:
        print(f"[FAD] VGGish unavailable ({e!r}).")
        print("[FAD] Falling back to the log-mel-stats encoder (fallback "
              "encoder, not comparable to VGGish-FAD); results are flagged.")
        return LogMelFallbackEncoder(device), True


def frechet_distance(mu1, sig1, mu2, sig2) -> float:
    """Fréchet distance between two Gaussians = FAD (Heusel/Kilgour)."""
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean = sqrtm(sig1 @ sig2)
    if isinstance(covmean, tuple):   # older scipy returns (matrix, error estimate)
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fad = diff @ diff + np.trace(sig1 + sig2 - 2.0 * covmean)
    return float(max(fad, 0.0))


def gaussian_stats(embs: np.ndarray):
    """mean vector + covariance (with a 1e-6 ridge)."""
    mu = embs.mean(0)
    if embs.shape[0] < 2:
        sig = np.eye(embs.shape[1]) * 1e-6
    else:
        sig = np.cov(embs, rowvar=False) + np.eye(embs.shape[1]) * 1e-6
    return mu, sig


# 2. Generation (deterministic; DDSP -> waveform, CVAE -> Griffin-Lim waveform)

class GriffinLimInverter:
    """Invert the z-scored log1p mel back to a 48 kHz waveform.
    Deterministic (fixed seed + fixed iters), with the same MelSpectrogram
    params as dataset.py."""

    def __init__(self, mel_mean, mel_std, n_frames, device):
        self.mel_mean = mel_mean.to(device)     # (n_mels, 1)
        self.mel_std = mel_std.to(device)
        self.device = device
        self.inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=N_FFT // 2 + 1, n_mels=N_MELS, sample_rate=SR).to(device)
        self.gl = torchaudio.transforms.GriffinLim(
            n_fft=N_FFT, hop_length=HOP, power=2.0, n_iter=GL_ITERS).to(device)

    @torch.no_grad()
    def __call__(self, mel_z: torch.Tensor) -> torch.Tensor:
        """mel_z: (n_mels, n_frames) in z-scored log1p domain → wav (T,)."""
        g = torch.Generator(device="cpu").manual_seed(GL_SEED)
        torch.manual_seed(GL_SEED)  # GriffinLim seeds its init phase from global RNG
        mel_log1p = mel_z.to(self.device) * self.mel_std + self.mel_mean
        mel_pow = torch.expm1(mel_log1p).clamp(min=0.0)     # undo log1p
        spec = self.inv_mel(mel_pow)                        # (n_fft/2+1, T)
        wav = self.gl(spec)                                 # (T,)
        return wav.cpu()


def load_checkpoint_and_model(ckpt_dir, dataset, device, identity_override=None):
    """Load best_model.pt + rebuild the matching model. identity and
    best_metric fall back to per-split defaults when absent from the
    checkpoint config.

    identity is resolved as config['identity'], then the per-split default,
    then the override, but an override is honoured only if it matches the
    checkpoint's trained identity dimension (obj_embed rows). On a mismatch
    the dimension that fits the weights wins, with a warning."""
    ckpt_dir = Path(ckpt_dir)
    ck = torch.load(ckpt_dir / "best_model.pt", map_location="cpu",
                    weights_only=False)
    cfg = ck["config"]
    backbone = cfg["backbone"]
    ablation = cfg["ablation"]
    split = cfg["split"]
    # identity: read from config; defaults by split (train.py rule) when absent.
    identity = cfg.get("identity") or identity_override
    if identity is None:
        identity = "obj" if split == "within" else "material"
    # Infer the trained identity from the saved obj_embed row count, which
    # is authoritative over any config or override guess.
    emb_rows = ck["model_state_dict"]["obj_embed.weight"].shape[0]
    trained_identity = "material" if emb_rows == dataset.n_materials else "obj"
    if identity != trained_identity:
        print(f"    [identity] requested '{identity}' but checkpoint was trained "
              f"with '{trained_identity}' (obj_embed rows={emb_rows}); using "
              f"'{trained_identity}' to match the weights.")
        identity = trained_identity
    best_metric = ck.get("best_metric", cfg.get("best_metric",
                          "val_mrstft" if backbone == "ddsp" else "val_recon"))

    # Sample geometry from the dataset (must match training-time geometry)
    sample = dataset[0]
    n_mels = sample["mel"].shape[0]
    n_frames = sample["mel"].shape[1]
    fw_samples = sample["force_waveform"].shape[0]

    # Descriptor dim from config; defaults to 6 when absent.
    force_feature_dim = cfg.get("force_feature_dim", 6)
    if backbone == "ddsp":
        from model_ddsp import build_model
        sig_samples = dataset[0]["waveform"].shape[0] \
            if "waveform" in dataset[0] else n_frames * 32
        model = build_model(
            n_objects=dataset.n_objects, ablation_level=ablation,
            n_bands=cfg.get("n_bands", 2048),
            use_modal=(cfg.get("modal", "on") == "on"),
            signal_samples=sig_samples, force_waveform_samples=fw_samples,
            identity=identity, n_materials=dataset.n_materials,
            force_feature_dim=force_feature_dim)
    else:
        from model import ConditionalMelVAE
        model = ConditionalMelVAE(
            n_mels=n_mels, n_frames=n_frames,
            latent_dim=cfg.get("latent_dim", 32),
            n_objects=dataset.n_objects, ablation_level=ablation,
            force_waveform_samples=fw_samples,
            identity=identity, n_materials=dataset.n_materials,
            force_feature_dim=force_feature_dim)

    # strict=False tolerates only the force_curve_{mean,std} buffers being
    # absent from a DDSP checkpoint (they are train-split stats, recomputed
    # below, not learned weights). Any other missing/unexpected key is a real
    # mismatch and is raised.
    incompat = model.load_state_dict(ck["model_state_dict"], strict=False)
    allowed_missing = {"force_curve_mean", "force_curve_std"}
    bad_missing = set(incompat.missing_keys) - allowed_missing
    if bad_missing or incompat.unexpected_keys:
        raise RuntimeError(
            f"state_dict mismatch for {ckpt_dir.name}: "
            f"missing={sorted(bad_missing)} unexpected={incompat.unexpected_keys}")
    model.to(device).eval()
    meta = dict(backbone=backbone, ablation=ablation, split=split,
                identity=identity, best_metric=best_metric,
                seed=cfg.get("seed"), ckpt_dir=str(ckpt_dir),
                force_curve_from_checkpoint=(
                    "force_curve_mean" not in incompat.missing_keys))
    return model, meta


@torch.no_grad()
def generate_waveforms(model, meta, dataset, indices, device,
                       feat_stats, fw_stats, gl_inv):
    """Generate one deterministic waveform per test index. Returns
    dict idx -> {gen: (T,), gt: (T,), obj_id_raw, material_idx}."""
    id_key = "obj_id" if meta["identity"] == "obj" else "material_idx"
    feat_mean, feat_std = feat_stats
    out = {}
    for i in indices:
        s = dataset[i]
        ident = torch.tensor([s[id_key]], dtype=torch.long, device=device)
        ff = ((s["force_features"] - feat_mean) / feat_std
              ).unsqueeze(0).to(device)
        if meta["backbone"] == "ddsp":
            # DDSP receives the raw force window (model pools + z-scores internally)
            fw = s["force_waveform"].unsqueeze(0).to(device)
            model.eval()
            wav, _ = model(ident, ff if meta["ablation"] in (2, 3, 4) else None,
                           fw if meta["ablation"] in (4, 5) else None,
                           loop_offsets=torch.zeros(1, dtype=torch.long,
                                                    device=device))
            gen = wav[0].cpu()
        else:
            # CVAE: force window z-scored with scalar train stats
            fwm, fws = fw_stats
            fw = ((s["force_waveform"] - fwm) / fws).unsqueeze(0).to(device)
            mel = model.generate(
                ident, ff if meta["ablation"] in (2, 3, 4) else None,
                fw if meta["ablation"] in (4, 5) else None, n_samples=1)[0]
            gen = gl_inv(mel)
        # Ground truth onset waveform (the window dataset.py built). For CVAE
        # GT goes through the same GL pipeline, so gen and gt live in the same
        # domain. DDSP GT = raw wav.
        if meta["backbone"] == "ddsp":
            gt = s["waveform"].cpu()
        else:
            gt = gl_inv(s["mel"])
        out[i] = dict(gen=gen, gt=gt,
                      obj_id_raw=s["obj_id_raw"],
                      material_idx=s["material_idx"])
    return out


# 3. Secondary metrics (onset error, attack log-mel L1 with shift-aligned
#    control, band-energy L1; all aggregated object-first)

_mel_metric = None
def _get_mel_metric(device):
    """Full-window mel (matches dataset params) for E-L1."""
    global _mel_metric
    if _mel_metric is None:
        _mel_metric = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            power=2.0).to(device)
    return _mel_metric


_attack_mel = None
def _get_attack_mel(device):
    """Small-FFT mel for the 10 ms (480-sample) attack window; the full
    n_fft=2048 exceeds 480 samples and cannot be reflect-padded."""
    global _attack_mel
    if _attack_mel is None:
        _attack_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=256, hop_length=64, n_mels=64,
            power=2.0).to(device)
    return _attack_mel


def onset_index(wav: torch.Tensor) -> int:
    """Onset = first sample whose running energy exceeds 10% of peak energy."""
    env = wav.abs()
    peak = env.max()
    if peak < 1e-9:
        return 0
    thr = 0.1 * peak
    above = (env > thr).nonzero()
    return int(above[0].item()) if above.numel() else 0


def logmel_l1_attack(gen, gt, device, align_shift=0):
    """First-10-ms log-mel L1. align_shift shifts gen by N samples before
    comparing (used for the shift-invariant baseline)."""
    n = int(ATTACK_WIN_MS / 1000 * SR)
    g = gen
    if align_shift != 0:
        if align_shift > 0:
            g = torch.cat([torch.zeros(align_shift), gen])[:len(gen)]
        else:
            g = torch.cat([gen[-align_shift:], torch.zeros(-align_shift)])
    mel = _get_attack_mel(device)
    with torch.no_grad():
        mg = torch.log1p(mel(g[:n].to(device)))
        mt = torch.log1p(mel(gt[:n].to(device)))
    return float((mg - mt).abs().mean().cpu())


def secondary_per_sample(gen, gt, device):
    """Per-sample secondary metrics, including the shift-aligned control."""
    on_g = onset_index(gen)
    on_t = onset_index(gt)
    onset_err_ms = abs(on_g - on_t) / SR * 1000.0
    onset_hit = onset_err_ms <= ONSET_TOL_MS
    attack_raw = logmel_l1_attack(gen, gt, device, align_shift=0)
    # Shift-aligned control: align gen's onset to gt's onset, then score.
    attack_aligned = logmel_l1_attack(gen, gt, device, align_shift=on_t - on_g)
    return dict(onset_err_ms=onset_err_ms, onset_hit=float(onset_hit),
                attack_logmel_l1=attack_raw,
                attack_logmel_l1_aligned=attack_aligned)


def energy_l1(gen, gt, device):
    """E-L1: L1 between log-mel-band energy envelopes (full 200 ms window)."""
    mel = _get_mel_metric(device)
    with torch.no_grad():
        mg = torch.log1p(mel(gen.to(device)))
        mt = torch.log1p(mel(gt.to(device)))
    return float((mg - mt).abs().mean().cpu())


# 4. Nearest-force retrieval reference

def build_nn_retriever(dataset, train_idx):
    """For each train sample, cache its L4 force waveform (the retrieval key)
    and its GT onset waveform (the retrieved audio). At eval, a test sample
    retrieves the train audio whose force waveform is nearest (L2 on the raw
    force window)."""
    keys = torch.stack([dataset[i]["force_waveform"] for i in train_idx])  # (N,S)
    keys = keys / (keys.norm(dim=1, keepdim=True) + 1e-9)                  # cosine-ish
    train_gt = {j: dataset[train_idx[j]] for j in range(len(train_idx))}
    return keys, train_idx, train_gt


def nn_retrieve(dataset, test_idx, keys, train_idx, gl_inv, backbone):
    """Return {test_idx -> retrieved GT waveform} for the nearest-force
    retrieval reference."""
    out = {}
    for i in test_idx:
        q = dataset[i]["force_waveform"]
        q = q / (q.norm() + 1e-9)
        sims = keys @ q                              # (N,) cosine similarity
        j = int(sims.argmax().item())
        s = dataset[train_idx[j]]
        if backbone == "ddsp":
            out[i] = s["waveform"]
        else:
            out[i] = gl_inv(s["mel"])
    return out


# 5. Object-level FAD, paired bootstrap over objects, effect sizes, Holm

def per_object_fad(gen_embs, gt_embs, obj_ids):
    """object_first_perobject_fad: for each object compute the FAD
    between that object's generated-embedding set and its GT-embedding set,
    then return {obj -> fad}. Objects with <2 impacts get a degenerate cov
    (ridge only), flagged as small-sample upstream."""
    from collections import defaultdict
    g_by, t_by = defaultdict(list), defaultdict(list)
    for e, o in zip(gen_embs, obj_ids):
        g_by[o].append(e)
    for e, o in zip(gt_embs, obj_ids):
        t_by[o].append(e)
    fad = {}
    for o in g_by:
        g = np.stack(g_by[o]); t = np.stack(t_by[o])
        mu_g, sg = gaussian_stats(g)
        mu_t, st = gaussian_stats(t)
        fad[o] = frechet_distance(mu_g, sg, mu_t, st)
    return fad


def paired_bootstrap_objects(delta_by_obj, n_boot=N_BOOTSTRAP, seed=0):
    """delta_by_obj: {obj -> (score_Li - score_L1)}. Resample objects with
    replacement (bootstrap unit = objects, not impacts). Return
    mean, (lo, hi) 95% CI, and the bootstrap distribution."""
    objs = sorted(delta_by_obj.keys())
    vals = np.array([delta_by_obj[o] for o in objs])
    rng = np.random.default_rng(seed)
    n = len(objs)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = vals[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(vals.mean()), (float(lo), float(hi)), boot


def cohens_d(vals):
    """Paired Cohen's d = mean(delta) / sd(delta)."""
    vals = np.asarray(vals)
    sd = vals.std(ddof=1)
    return float(vals.mean() / sd) if sd > 1e-12 else 0.0


def cliffs_delta(vals):
    """Cliff's delta of the paired deltas vs 0 = P(d>0) - P(d<0)."""
    vals = np.asarray(vals)
    return float((np.sum(vals > 0) - np.sum(vals < 0)) / len(vals))


def bootstrap_p_two_sided(boot):
    """Two-sided bootstrap p for H0: mean delta = 0 (proportion of the
    bootstrap distribution on the far side of 0, doubled)."""
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(min(p, 1.0))


def holm_correction(pvals: dict) -> dict:
    """Holm–Bonferroni over the primary family. pvals: {level -> p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for rank, (lvl, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        adj = max(adj, prev)   # enforce monotonicity
        out[lvl] = adj
        prev = adj
    return out


# 6. Driver

def embed_all(encoder, wavs_by_idx, key):
    """Embed the `key` waveform of every entry → (embs array, obj order)."""
    idxs = sorted(wavs_by_idx.keys())
    embs = np.stack([encoder.embed(wavs_by_idx[i][key]) for i in idxs])
    objs = [wavs_by_idx[i]["obj_id_raw"] for i in idxs]
    mats = [wavs_by_idx[i]["material_idx"] for i in idxs]
    return idxs, embs, objs, mats


def eval_one_model(ckpt_dir, dataset, split_idx, encoder, device, gl_inv,
                   feat_stats, fw_stats, nn_bundle, eval_split_name,
                   identity_override=None):
    """Evaluate a single (ablation, seed) checkpoint on the chosen split.
    Returns per-object FAD, per-object secondary means, and per-sample raw
    material tags for downstream aggregation."""
    train_idx, val_idx, test_idx = split_idx
    idx = {"test": test_idx, "val": val_idx, "dev": val_idx}[eval_split_name]

    model, meta = load_checkpoint_and_model(
        ckpt_dir, dataset, device, identity_override=identity_override)
    # DDSP L4: if the checkpoint lacks the force_curve buffers, recompute
    # them from the train split so the force curve is z-scored exactly as
    # at train time.
    if (meta["backbone"] == "ddsp" and meta["ablation"] in (4, 5)
            and not meta["force_curve_from_checkpoint"]):
        fw = torch.stack([dataset[i]["force_waveform"] for i in train_idx])
        curves = model.pool_force_curve(fw, model.force_frames, model.w)
        flat = curves.reshape(-1, 2)
        model.set_force_curve_stats(flat.mean(0).to(device),
                                    flat.std(0).clamp(min=1e-8).to(device))
    gen = generate_waveforms(model, meta, dataset, idx, device,
                             feat_stats, fw_stats, gl_inv)

    # Embeddings for FAD (object-first)
    idxs = sorted(gen.keys())
    gen_embs = np.stack([encoder.embed(gen[i]["gen"]) for i in idxs])
    gt_embs = np.stack([encoder.embed(gen[i]["gt"]) for i in idxs])
    objs = [gen[i]["obj_id_raw"] for i in idxs]
    mats = [gen[i]["material_idx"] for i in idxs]

    fad_by_obj = per_object_fad(gen_embs, gt_embs, objs)

    # Secondary metrics, per sample → aggregate object-first
    from collections import defaultdict
    sec_acc = defaultdict(lambda: defaultdict(list))
    for i in idxs:
        s = secondary_per_sample(gen[i]["gen"], gen[i]["gt"], device)
        el1 = energy_l1(gen[i]["gen"], gen[i]["gt"], device)
        o = gen[i]["obj_id_raw"]
        for k, v in s.items():
            sec_acc[k][o].append(v)
        sec_acc["energy_l1"][o].append(el1)
    sec_by_obj = {k: {o: float(np.mean(vs)) for o, vs in d.items()}
                  for k, d in sec_acc.items()}

    obj_to_mat = {gen[i]["obj_id_raw"]: gen[i]["material_idx"] for i in idxs}
    return dict(meta=meta, fad_by_obj=fad_by_obj, sec_by_obj=sec_by_obj,
                obj_to_mat=obj_to_mat)


def nn_lower_bound_fad(dataset, split_idx, encoder, gl_inv, backbone,
                       nn_bundle, gen_gt_embs_by_obj):
    """Nearest-force retrieval reference: object-level FAD of retrieved train
    audio vs GT, i.e. the FAD obtained by just retrieving the train clip with
    the nearest force waveform."""
    train_idx, val_idx, test_idx = split_idx
    keys, tr_idx, _ = nn_bundle
    retrieved = nn_retrieve(dataset, test_idx, keys, tr_idx, gl_inv, backbone)
    # GT embeddings must match the same test objects
    objs, ret_embs, gt_embs = [], [], []
    for i in test_idx:
        objs.append(dataset[i]["obj_id_raw"])
        ret_embs.append(encoder.embed(retrieved[i]))
        # GT waveform for the test sample
        if backbone == "ddsp":
            gt = dataset[i]["waveform"]
        else:
            gt = gl_inv(dataset[i]["mel"])
        gt_embs.append(encoder.embed(gt))
    fad_by_obj = per_object_fad(np.stack(ret_embs), np.stack(gt_embs), objs)
    return float(np.mean(list(fad_by_obj.values())))


def aggregate_ablation_family(models_by_level, encoder, dataset, split_idx,
                              is_fallback):
    """Given {level -> [per-seed eval dicts]}, compute the primary object-first
    FAD, paired bootstrap deltas vs L1, effect sizes, Holm p, and per-material
    breakdown. Seeds are a random effect: per seed compute the object-first FAD,
    then report mean±CI across seeds and the pooled per-object bootstrap."""
    results = {"fad_encoder": encoder.name,
               "fad_encoder_compliant": (not is_fallback),
               "fad_aggregation": "object_first_perobject_fad",
               "levels": {}}

    # Primary FAD per level: object-first mean, across seeds as random effect.
    level_obj_fad = {}   # level -> {obj -> [fad per seed]}
    from collections import defaultdict
    for lvl, evals in models_by_level.items():
        acc = defaultdict(list)
        for ev in evals:
            for o, f in ev["fad_by_obj"].items():
                acc[o].append(f)
        # average over seeds per object (random effect)
        level_obj_fad[lvl] = {o: float(np.mean(v)) for o, v in acc.items()}
        obj_means = np.array(list(level_obj_fad[lvl].values()))
        results["levels"][lvl] = {
            "object_first_fad_mean": float(obj_means.mean()),
            "object_first_fad_across_object_sd": float(obj_means.std(ddof=1))
            if len(obj_means) > 1 else 0.0,
            "n_objects": int(len(obj_means)),
            "n_seeds": len(evals),
        }

    # Paired bootstrap: L2/L3/L4 vs L1, over objects.
    if 1 not in level_obj_fad:
        results["primary_comparisons"] = {
            "error": "L1 checkpoint missing; cannot compute Li-L1 deltas."}
        return results, level_obj_fad

    l1 = level_obj_fad[1]
    comparisons, raw_p = {}, {}
    for lvl in PRIMARY_FAMILY:
        if lvl not in level_obj_fad:
            continue
        li = level_obj_fad[lvl]
        common = sorted(set(l1) & set(li))
        # delta = Li - L1 (negative = Li better, since FAD lower is better)
        delta = {o: li[o] - l1[o] for o in common}
        mean, ci, boot = paired_bootstrap_objects(delta, seed=lvl)
        vals = np.array([delta[o] for o in common])
        p = bootstrap_p_two_sided(boot)
        raw_p[lvl] = p
        comparisons[lvl] = {
            "delta_Li_minus_L1_mean": mean,
            "ci95": ci,
            "cohens_d": cohens_d(vals),
            "cliffs_delta": cliffs_delta(vals),
            "p_raw": p,
            "n_objects_paired": len(common),
            "interpretation": _interpret(mean, ci),
        }
    holm = holm_correction(raw_p)
    for lvl in comparisons:
        c = comparisons[lvl]
        c["p_holm"] = holm[lvl]
        c["significant_holm_0.05"] = bool(holm[lvl] < 0.05)
        # Decision rule: a level counts as helping (or harming) when
        # |Cohen's d| >= 0.2 and the Holm-corrected p < 0.05; the sign of the
        # mean difference says which. Otherwise distinguish "no detected
        # difference" (CI includes 0 and |d| < 0.2) from "inconclusive".
        d_ok = abs(c["cohens_d"]) >= 0.2
        p_ok = c["significant_holm_0.05"]
        lo, hi = c["ci95"]
        ci_excludes_0 = (hi < 0) or (lo > 0)
        if d_ok and p_ok and c["delta_Li_minus_L1_mean"] < 0:
            c["verdict"] = "FORCE_HELPS"
        elif d_ok and p_ok and c["delta_Li_minus_L1_mean"] > 0:
            c["verdict"] = "FORCE_HARMS"
        elif (not ci_excludes_0) and abs(c["cohens_d"]) < 0.2:
            c["verdict"] = "NO_DETECTED_DIFFERENCE"
        else:
            c["verdict"] = "INCONCLUSIVE"
    results["primary_comparisons"] = comparisons

    # Per-material FAD: CI only for materials with n_obj >= 3; else qualitative.
    obj_to_mat = {}
    for evals in models_by_level.values():
        for ev in evals:
            obj_to_mat.update(ev["obj_to_mat"])
    per_mat = {}
    for lvl in level_obj_fad:
        mat_acc = defaultdict(list)
        for o, f in level_obj_fad[lvl].items():
            mat_acc[obj_to_mat.get(o, -1)].append(f)
        per_mat[lvl] = {}
        for mat_idx, vals in mat_acc.items():
            name = MATERIALS[mat_idx] if 0 <= mat_idx < len(MATERIALS) else "unknown"
            entry = {"n_objects": len(vals),
                     "object_first_fad_mean": float(np.mean(vals))}
            if len(vals) >= PER_MATERIAL_MIN_OBJ:
                m, ci, _ = paired_bootstrap_objects(
                    {i: v for i, v in enumerate(vals)}, seed=1)
                entry["ci95"] = ci
                entry["qualitative_only"] = False
            else:
                entry["qualitative_only"] = True  # Iron/Glass etc.
            per_mat[lvl][name] = entry
    results["per_material"] = per_mat
    return results, level_obj_fad


def _interpret(mean, ci):
    """One-line reading of a FAD delta (lower is better, so a negative delta
    means the force level improved on the identity baseline)."""
    lo, hi = ci
    if hi < 0:
        return "Li better than L1 (CI excludes 0)"
    if lo > 0:
        return "Li worse than L1 (CI excludes 0)"
    if lo < 0 < hi:
        return ("CI includes 0: no difference detected at this granularity; "
                "the CI width bounds what the design could have detected")
    return "inconclusive"


def secondary_summary(models_by_level):
    """Object-first secondary metrics per level plus the shift-aligned
    control (raw attack vs aligned attack). Reported as supporting evidence
    only."""
    from collections import defaultdict
    out = {}
    for lvl, evals in models_by_level.items():
        acc = defaultdict(list)
        for ev in evals:
            for metric, by_obj in ev["sec_by_obj"].items():
                acc[metric].extend(by_obj.values())
        out[lvl] = {m: {"object_first_mean": float(np.mean(v))}
                    for m, v in acc.items()}
        # shift-invariance check: aligned should not be dramatically better,
        # else the raw score was just measuring misalignment.
        if "attack_logmel_l1" in out[lvl] and "attack_logmel_l1_aligned" in out[lvl]:
            raw = out[lvl]["attack_logmel_l1"]["object_first_mean"]
            al = out[lvl]["attack_logmel_l1_aligned"]["object_first_mean"]
            out[lvl]["shift_invariance_gap"] = {
                "raw_minus_aligned": raw - al,
                "note": ("a large positive gap indicates the raw attack score "
                         "is inflated by misalignment; a small gap indicates "
                         "the score is robust to alignment.")}
    return out


# 7. CLI + main

def discover_checkpoints(ckpt_args, backbone, split):
    """Expand --ckpt args into {ablation_level -> [dirs (one per seed)]}.
    Each arg is a directory that either is a run dir (contains best_model.pt)
    or a parent holding several run dirs. Ablation level is read from each
    run's config.json."""
    dirs = []
    for a in ckpt_args:
        p = Path(a)
        if (p / "best_model.pt").exists():
            dirs.append(p)
        else:
            dirs.extend(sorted(d for d in p.glob("*") if (d / "best_model.pt").exists()))
    by_level = {}
    for d in dirs:
        cfg = json.load(open(d / "config.json"))
        if cfg["backbone"] != backbone or cfg["split"] != split:
            continue
        by_level.setdefault(cfg["ablation"], []).append(d)
    return by_level


def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate trained checkpoints (object-level FAD, paired "
                    "bootstrap, effect sizes, secondary metrics).")
    ap.add_argument("--backbone", default="ddsp", choices=["ddsp", "cvae"],
                    help="model backbone (default: ddsp)")
    ap.add_argument("--split", default="within", choices=["within", "heldout"],
                    help="within = seen objects; heldout = unseen objects")
    ap.add_argument("--identity", default=None, choices=["obj", "material"],
                    help="override identity embedding; default = checkpoint config")
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="run dir(s) or parent dir(s) holding best_model.pt "
                         "(one per ablation level x seed)")
    ap.add_argument("--dev", action="store_true",
                    help="evaluate on the validation split instead of test")
    ap.add_argument("--out", default=None, help="output dir (default results/)")
    ap.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    return ap.parse_args()


def main():
    args = parse_args()
    eval_split = "dev" if args.dev else "test"

    if not args.dev:
        print("\nEvaluating on the test split (use --dev for validation).\n")
    else:
        print("\n[--dev] Evaluating on the validation split, not test.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                          if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset (built over all objects; DDSP needs the waveform target too)
    dataset = ObjectFolderRealDataset(return_waveform=(args.backbone == "ddsp"))
    split_file = SPLITS_DIR / f"splits_{args.split}.json"
    split_idx = load_split_indices(dataset, str(split_file))
    train_idx, val_idx, test_idx = split_idx

    # Train-split stats (mel z-score, force stats), recomputed here so the
    # script is self-contained; must match training-time stats.
    mel_mean, mel_std = compute_mel_stats(dataset, train_idx)
    dataset.set_mel_stats(mel_mean, mel_std)
    feat = torch.stack([dataset[i]["force_features"] for i in train_idx])
    feat_mean = feat.mean(0); feat_std = feat.std(0); feat_std[feat_std < 1e-6] = 1.0
    fwv = torch.stack([dataset[i]["force_waveform"] for i in train_idx])
    fw_stats = (fwv.mean(), fwv.std().clamp(min=1e-6))

    n_frames = dataset[0]["mel"].shape[1]
    gl_inv = GriffinLimInverter(mel_mean, mel_std, n_frames, device)

    encoder, is_fallback = build_fad_encoder(device)

    by_level = discover_checkpoints(args.ckpt, args.backbone, args.split)
    if not by_level:
        raise SystemExit(f"No checkpoints matched backbone={args.backbone} "
                         f"split={args.split} in {args.ckpt}")
    print(f"Checkpoints by level: "
          f"{ {k: len(v) for k, v in sorted(by_level.items())} }")

    # Evaluate every (level, seed) checkpoint
    models_by_level = {}
    for lvl, dirs in sorted(by_level.items()):
        evals = []
        for d in dirs:
            print(f"  eval L{lvl} :: {d.name}")
            ev = eval_one_model(d, dataset, split_idx, encoder, device, gl_inv,
                                (feat_mean, feat_std), fw_stats, None, eval_split,
                                identity_override=args.identity)
            evals.append(ev)
        models_by_level[lvl] = evals

    # Primary aggregation + bootstrap + Holm
    global N_BOOTSTRAP
    N_BOOTSTRAP = args.n_bootstrap
    results, level_obj_fad = aggregate_ablation_family(
        models_by_level, encoder, dataset, split_idx, is_fallback)
    results["secondary"] = secondary_summary(models_by_level)

    # Nearest-force retrieval reference (uses the backbone's generation domain)
    try:
        nn_bundle = build_nn_retriever(dataset, train_idx)
        nn_fad = nn_lower_bound_fad(dataset, split_idx, encoder, gl_inv,
                                    args.backbone, nn_bundle, None)
        results["nn_lower_bound_object_first_fad"] = nn_fad
    except Exception as e:
        results["nn_lower_bound_object_first_fad"] = f"failed: {e!r}"

    results["meta"] = {
        "backbone": args.backbone, "split": args.split,
        "eval_split": eval_split, "dev_mode": args.dev,
        "n_bootstrap": args.n_bootstrap,
        "onset_tol_ms": ONSET_TOL_MS, "attack_win_ms": ATTACK_WIN_MS,
        "seeds_per_level": {k: len(v) for k, v in models_by_level.items()},
        "encoder_note": ("fallback encoder, not comparable to VGGish-FAD"
                         if is_fallback else "VGGish primary encoder"),
    }

    # Write machine-readable + human-readable outputs
    out_dir = Path(args.out) if args.out else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.backbone}_{args.split}_{eval_split}"
    json_path = out_dir / f"results_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(_jsonify(results), f, indent=2)
    md_path = out_dir / f"summary_{tag}.md"
    with open(md_path, "w") as f:
        f.write(render_markdown(results))
    print(f"\nWrote:\n  {json_path}\n  {md_path}")
    print("\n" + render_markdown(results))


def _jsonify(o):
    if isinstance(o, dict):
        return {str(k): _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def render_markdown(r):
    L = []
    enc = r.get("fad_encoder", "?")
    ok = r.get("fad_encoder_compliant", False)
    L.append(f"# Evaluation: {r['meta']['backbone']} / "
             f"{r['meta']['split']} / {r['meta']['eval_split']}\n")
    if r["meta"].get("dev_mode"):
        L.append("> Validation split (--dev), not test.\n")
    L.append(f"- FAD encoder: **{enc}** "
             f"({'primary encoder' if ok else 'fallback encoder, not comparable to VGGish-FAD'})")
    L.append(f"- Aggregation: {r.get('fad_aggregation')}")
    L.append(f"- Bootstrap: paired over objects, n={r['meta']['n_bootstrap']}")
    nn = r.get("nn_lower_bound_object_first_fad")
    L.append(f"- Nearest-force retrieval reference (object-first FAD): {nn}\n")

    L.append("## Primary: object-first FAD per level\n")
    L.append("| Level | FAD (obj-first mean) | across-obj sd | n_obj | n_seeds |")
    L.append("|---|---|---|---|---|")
    for lvl in sorted(r["levels"]):
        d = r["levels"][lvl]
        L.append(f"| L{lvl} | {d['object_first_fad_mean']:.4f} | "
                 f"{d['object_first_fad_across_object_sd']:.4f} | "
                 f"{d['n_objects']} | {d['n_seeds']} |")
    L.append("")

    pc = r.get("primary_comparisons", {})
    if "error" in pc:
        L.append(f"**Primary comparisons unavailable: {pc['error']}**\n")
    else:
        L.append("## Primary comparisons: Li − L1 (negative = force helps)\n")
        L.append("| Cmp | Δ mean | 95% CI | Cohen d | Cliff δ | p_raw | p_Holm | sig | decision |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for lvl in sorted(pc):
            c = pc[lvl]
            L.append(f"| L{lvl}−L1 | {c['delta_Li_minus_L1_mean']:.4f} | "
                     f"[{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}] | "
                     f"{c['cohens_d']:.3f} | {c['cliffs_delta']:.3f} | "
                     f"{c['p_raw']:.4f} | {c['p_holm']:.4f} | "
                     f"{'YES' if c['significant_holm_0.05'] else 'no'} | "
                     f"**{c['verdict']}** |")
        L.append("\n> Decision rule: FORCE_HELPS / FORCE_HARMS need |d| >= 0.2 and "
                 "Holm p < 0.05, with the sign of Δ giving the direction. "
                 "NO_DETECTED_DIFFERENCE = CI includes 0 and |d| < 0.2. "
                 "INCONCLUSIVE = wide or mixed CI.\n")

    sec = r.get("secondary", {})
    if sec:
        L.append("## Secondary metrics (supporting only): object-first means\n")
        L.append("| Level | onset_err_ms | onset_hit@5ms | attack_L1 | attack_L1_aligned | E-L1 | shift_gap |")
        L.append("|---|---|---|---|---|---|---|")
        for lvl in sorted(sec):
            s = sec[lvl]
            def g(k): return s.get(k, {}).get("object_first_mean", float("nan"))
            gap = s.get("shift_invariance_gap", {}).get("raw_minus_aligned", float("nan"))
            L.append(f"| L{lvl} | {g('onset_err_ms'):.3f} | {g('onset_hit'):.3f} | "
                     f"{g('attack_logmel_l1'):.4f} | {g('attack_logmel_l1_aligned'):.4f} | "
                     f"{g('energy_l1'):.4f} | {gap:.4f} |")
        L.append("\n> shift_gap = raw − aligned attack L1. A large positive gap "
                 "means the raw score was inflated by misalignment; a small gap "
                 "means the attack score is not just measuring alignment.\n")

    pm = r.get("per_material", {})
    if pm:
        L.append("## Per-material FAD (CI only for n_obj >= 3; Iron/Glass qualitative)\n")
        for lvl in sorted(pm):
            L.append(f"### L{lvl}")
            L.append("| Material | n_obj | FAD | qualitative-only |")
            L.append("|---|---|---|---|")
            for mat, d in sorted(pm[lvl].items()):
                L.append(f"| {mat} | {d['n_objects']} | "
                         f"{d['object_first_fad_mean']:.4f} | "
                         f"{'YES' if d['qualitative_only'] else 'no'} |")
            L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
