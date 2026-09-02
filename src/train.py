"""Training script for the force-conditioned impact-sound models.

Trains either the CVAE backbone (mel-spectrogram target, model.py) or the
DDSP backbone (waveform target, model_ddsp.py) at one of four
force-conditioning levels, selected with --ablation:
  1  identity embedding only
  2  + peak-force scalar
  3  + 6-D force descriptor
  4  + time-resolved force frames (the 200 ms force window)
on either the within split (seen objects, per-object 70/15/15) or the
held-out split (unseen objects, material-stratified), loaded from
splits/splits_<split>.json (both files ship in splits/; make_splits.py
regenerates them with seed 42).

All normalisation statistics are computed on the train split of the
chosen split and cached under stats/, so that train/val/test and all four
levels share them: the per-mel-bin z-score (mel_stats_<split>.pt), the
scalar force-waveform z-score used by the CVAE L4 encoder
(force_wav_stats_<split>.pt), the per-channel [RMS, max-abs] force-curve
z-score used by the DDSP L4 path (force_curve_stats_<split>.pt) and the
global waveform scale of the DDSP target (wav_scale_<split>.pt). The 6-D
descriptor is z-scored with train-split statistics saved next to the
checkpoint (feature_stats.pt). No per-sample peak normalisation is used
anywhere.

The CVAE is trained with MSE + beta-KL (cyclical annealing, free bits);
the DDSP backbone with a multi-scale STFT loss on the waveform. The best
checkpoint is selected on validation reconstruction error (CVAE) or
validation MRSTFT (DDSP). Checkpoints, config and history are written to
experiments/ablation_<N>_<split>_<backbone> unless --save_dir is given.

Training-side variants (the Table 2 rows):
  --env_lambda W   DDSP only: add an envelope-L1 auxiliary loss
                   (W * L1 between the RMS envelopes of prediction and
                   target; signal_metrics.rms_env definition). The
                   validation loss, validation metric and best-model
                   selection stay plain MRSTFT.
  --init_from P    initialise the weights from another run's best_model.pt
                   (P = the file or its run directory). Tensors whose
                   shapes differ from the new model are skipped, as are
                   the force-curve statistics buffers (they are recomputed
                   on the current train split by set_force_curve_stats).
  --synthetic      train on the synthetic pretraining corpus
                   (synthetic_data.py: noise bursts convolved with room
                   IRs from the pool built by make_synthetic_ir_pool.py)
                   instead of ObjectFolder-Real. The training portion is
                   redrawn every epoch.

Usage:
    python3 src/train.py --backbone ddsp --ablation 4 --split within --seed 42
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from dataset import (ObjectFolderRealDataset, DATA_ROOT,
                     compute_mel_stats, load_split_indices, FEATURE_SETS_V2)
from model import ConditionalMelVAE, cvae_loss

# repository-relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "splits"
STATS_DIR = PROJECT_ROOT / "stats"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the force-conditioned generators (CVAE or DDSP)")
    parser.add_argument("--ablation", type=int, default=3, choices=[1, 2, 3, 4],
                        help="Force-conditioning level: 1=identity only, 2=+peak-force "
                             "scalar, 3=+6-D descriptor, 4=+time-resolved force frames")
    parser.add_argument("--feature_set", type=str, default="v1",
                        choices=["v1", "impulse", "v2A", "v2AB", "pca6"],
                        help="Force-feature set: v1=original 6-D descriptor; "
                             "impulse=L2' impulse scalar; v2A=L3'_A "
                             "[peak, contact_dur, asymmetry]; v2AB=L3'_AB (6-D); "
                             "pca6=6 PCA scores on a train-split basis "
                             "(stats/pca6_basis_<split>.pt, fit_pca_basis.py). "
                             "L2 uses dim 0 of the vector.")
    parser.add_argument("--backbone", type=str, default="cvae",
                        choices=["cvae", "ddsp"],
                        help="cvae=mel-spectrogram CVAE (model.py); "
                             "ddsp=filterbank-DDSP hybrid (model_ddsp.py, "
                             "waveform target + MRSTFT loss)")
    parser.add_argument("--n_bands", type=int, default=2048,
                        help="[ddsp] number of pre-baked noise bands "
                             "(quick check: 256)")
    parser.add_argument("--modal", type=str, default="on", choices=["on", "off"],
                        help="[ddsp] damped-sinusoid modal branch (NBN+modal "
                             "variant, default on)")
    parser.add_argument("--split", type=str, default="heldout",
                        choices=["heldout", "within"],
                        help="heldout=object-held-out (unseen objects), "
                             "within=within-object 70/15/15 (seen objects)")
    parser.add_argument("--identity", type=str, default=None,
                        choices=["obj", "material"],
                        help="Identity embedding. Default by split: "
                             "within->obj, heldout->material (held-out objects "
                             "are unseen, so a per-object row would be "
                             "untrained; material generalises). Same for all "
                             "4 ablation levels.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--beta_max", type=float, default=0.1,
                        help="Max KL weight (cyclical annealing peak)")
    parser.add_argument("--beta_cycle", type=int, default=100,
                        help="Epochs per cyclical annealing cycle")
    parser.add_argument("--beta_ratio", type=float, default=0.5,
                        help="Fraction of cycle spent increasing beta (rest is held)")
    parser.add_argument("--free_bits", type=float, default=0.25,
                        help="Free bits per latent dim (0=disable)")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--split_file", type=str, default=None,
                        help="Override split JSON path (default: "
                             "<project>/splits/splits_<split>.json)")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env_lambda", type=float, default=0.0,
                        help="[ddsp] Weight of the envelope-L1 auxiliary loss "
                             "(0 = off). Validation loss/metric and best-model "
                             "selection stay plain MRSTFT.")
    parser.add_argument("--env_win", type=int, default=240,
                        help="[ddsp] RMS envelope window for the env loss")
    parser.add_argument("--env_hop", type=int, default=120,
                        help="[ddsp] RMS envelope hop for the env loss")
    parser.add_argument("--init_from", type=str, default=None,
                        help="Initialise weights from another run's "
                             "best_model.pt (file or run directory); "
                             "shape-mismatched tensors and the force-curve "
                             "statistics buffers are skipped")
    parser.add_argument("--synthetic", action="store_true",
                        help="Train on the synthetic pretraining corpus "
                             "(synthetic_data.py) instead of ObjectFolder-Real")
    parser.add_argument("--ir_pool", type=str, default=None,
                        help="IR pool for --synthetic (default: "
                             "cache/synthetic_ir_pool.pt, built by "
                             "make_synthetic_ir_pool.py)")
    parser.add_argument("--synthetic_base_seed", type=int, default=4243,
                        help="Base seed of the synthetic corpus (samples are "
                             "deterministic given base seed, epoch, index)")
    return parser.parse_args()


def get_beta(epoch, beta_max, beta_cycle, beta_ratio):
    """Cyclical annealing (Fu et al., 2019).

    Each cycle ramps linearly from 0 to beta_max over (ratio * cycle) epochs,
    then holds at beta_max for the remainder.
    """
    pos_in_cycle = epoch % beta_cycle
    ramp_epochs = int(beta_cycle * beta_ratio)
    if pos_in_cycle < ramp_epochs:
        return beta_max * pos_in_cycle / ramp_epochs
    return beta_max


def normalize_features(train_features):
    """Compute feature-wise mean/std from training set for normalization."""
    feats = torch.stack(train_features)
    mean = feats.mean(dim=0)
    std = feats.std(dim=0)
    std[std < 1e-6] = 1.0
    return mean, std


def get_mel_stats(dataset, train_idx, split_name):
    """Load cached mel z-score stats, or compute from the train split and cache."""
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATS_DIR / f"mel_stats_{split_name}.pt"
    if path.exists():
        stats = torch.load(path)
        if stats.get('n_train') == len(train_idx):
            print(f"Loaded mel stats: {path}")
            return stats['mean'], stats['std']
        print(f"Stale mel stats ({stats.get('n_train')} != {len(train_idx)} "
              f"train samples); recomputing: {path}")
    mean, std = compute_mel_stats(dataset, train_idx)
    torch.save({'mean': mean, 'std': std, 'split': split_name,
                'n_train': len(train_idx)}, path)
    print(f"Saved mel stats: {path}")
    return mean, std


def get_force_wav_stats(dataset, train_idx, split_name):
    """Load cached force-waveform z-score stats (global scalar mean/std over
    the train split), or compute and cache."""
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATS_DIR / f"force_wav_stats_{split_name}.pt"
    if path.exists():
        stats = torch.load(path)
        if stats.get('n_train') == len(train_idx):
            print(f"Loaded force waveform stats: {path}")
            return stats['mean'], stats['std']
        print(f"Stale force waveform stats ({stats.get('n_train')} != "
              f"{len(train_idx)} train samples); recomputing: {path}")
    fw = torch.stack([dataset[i]['force_waveform'] for i in train_idx])
    mean = fw.mean()
    std = fw.std().clamp(min=1e-6)
    torch.save({'mean': mean, 'std': std, 'split': split_name,
                'n_train': len(train_idx)}, path)
    print(f"Saved force waveform stats: {path} "
          f"(mean={mean:.6f}, std={std:.6f})")
    return mean, std


def get_force_curve_stats(dataset, train_idx, split_name, force_frames, w):
    """[ddsp] Per-channel (RMS, max-abs) z-score stats of the L4 force curve
    over the train split. Pools each raw force window into a
    (force_frames, 2) curve, then takes the per-channel mean/std over all
    train frames. Cached in stats/force_curve_stats_<split>.pt; reused by all
    ablation levels. Returns (mean, std) each shape (2,)."""
    from model_ddsp import ForceConditionedFilterbankDDSP as _M
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATS_DIR / f"force_curve_stats_{split_name}.pt"
    if path.exists():
        stats = torch.load(path)
        if stats.get('n_train') == len(train_idx) and \
                stats.get('force_frames') == force_frames:
            print(f"Loaded force curve stats: {path}")
            return stats['mean'], stats['std']
        print(f"Stale force curve stats; recomputing: {path}")
    fw = torch.stack([dataset[i]['force_waveform'] for i in train_idx])  # (N,S)
    curves = _M.pool_force_curve(fw, force_frames, w)      # (N, F, 2)
    flat = curves.reshape(-1, 2)                           # (N*F, 2)
    mean = flat.mean(dim=0)                                # (2,)
    std = flat.std(dim=0).clamp(min=1e-8)                 # (2,)
    torch.save({'mean': mean, 'std': std, 'split': split_name,
                'n_train': len(train_idx), 'force_frames': force_frames}, path)
    print(f"Saved force curve stats: {path} "
          f"(mean={mean.tolist()}, std={std.tolist()})")
    return mean, std


def get_wav_scale(dataset, train_idx, split_name):
    """[ddsp] Global waveform scaling for the DDSP target: the scalar std of
    the raw onset-window waveform over the train split. Cached in
    stats/wav_scale_<split>.pt and reused by all ablation levels."""
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATS_DIR / f"wav_scale_{split_name}.pt"
    if path.exists():
        stats = torch.load(path)
        if stats.get('n_train') == len(train_idx):
            print(f"Loaded wav scale: {path} (std={stats['std']:.6f})")
            return stats['std']
        print(f"Stale wav scale ({stats.get('n_train')} != {len(train_idx)} "
              f"train samples); recomputing: {path}")
    wavs = torch.stack([dataset[i]['waveform'] for i in train_idx])
    std = wavs.std().clamp(min=1e-8)
    torch.save({'std': std, 'split': split_name, 'n_train': len(train_idx)}, path)
    print(f"Saved wav scale: {path} (std={std:.6f})")
    return std


def collate_fn(batch):
    """Custom collate to handle dict samples."""
    out = {
        'force_features': torch.stack([b['force_features'] for b in batch]),
        'force_waveform': torch.stack([b['force_waveform'] for b in batch]),
        'mel': torch.stack([b['mel'] for b in batch]),
        'obj_id': torch.tensor([b['obj_id'] for b in batch], dtype=torch.long),
        'material_idx': torch.tensor([b['material_idx'] for b in batch],
                                     dtype=torch.long),
        'striking_N': torch.tensor([b['striking_N'] for b in batch]),
    }
    if 'waveform' in batch[0]:  # only present when return_waveform=True (ddsp)
        out['waveform'] = torch.stack([b['waveform'] for b in batch])
    return out


def train_epoch(model, loader, optimizer, device, beta, free_bits,
                feat_mean, feat_std, fw_mean, fw_std, id_key):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0

    for batch in loader:
        mel = batch['mel'].to(device)
        obj_id = batch[id_key].to(device)
        force_feat = (batch['force_features'].to(device) - feat_mean) / feat_std
        force_wav = (batch['force_waveform'].to(device) - fw_mean) / fw_std

        recon, mu, logvar = model(mel, obj_id, force_feat, force_wav)
        loss, recon_loss, kl_loss = cvae_loss(recon, mel, mu, logvar, beta=beta, free_bits=free_bits)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


@torch.no_grad()
def eval_epoch(model, loader, device, beta, free_bits,
               feat_mean, feat_std, fw_mean, fw_std, id_key):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0

    for batch in loader:
        mel = batch['mel'].to(device)
        obj_id = batch[id_key].to(device)
        force_feat = (batch['force_features'].to(device) - feat_mean) / feat_std
        force_wav = (batch['force_waveform'].to(device) - fw_mean) / fw_std

        recon, mu, logvar = model(mel, obj_id, force_feat, force_wav)
        loss, recon_loss, kl_loss = cvae_loss(recon, mel, mu, logvar, beta=beta, free_bits=free_bits)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


# DDSP backbone epochs (waveform target + MRSTFT, optional envelope-L1 aux)

def _rms_env_batched(x, win, hop):
    """Batched counterpart of signal_metrics.rms_env (identical definition):
    x (B, T) -> RMS envelopes (B, n_frames), sqrt(mean(x^2) + 1e-12)."""
    pad = (win - hop) // 2
    x2 = nn.functional.pad(x ** 2, (pad, pad))
    fr = x2.unfold(-1, win, hop)
    return torch.sqrt(fr.mean(dim=-1) + 1e-12)


def _env_l1(pred, target, win, hop):
    """L1 between the RMS envelopes of pred and target (both already in the
    wav_scale-normalised domain)."""
    return (_rms_env_batched(pred, win, hop)
            - _rms_env_batched(target, win, hop)).abs().mean()


def _ddsp_batch_loss(model, loss_fn, batch, device,
                     feat_mean, feat_std, wav_scale, id_key,
                     env_lambda=0.0, env_win=240, env_hop=120):
    obj_id = batch[id_key].to(device)
    force_feat = (batch['force_features'].to(device) - feat_mean) / feat_std
    # The DDSP model takes the raw force window and does its own pooling and
    # per-channel z-scoring (set_force_curve_stats).
    force_wav = batch['force_waveform'].to(device)
    # Waveform target, scaled by the global train-split std
    target = batch['waveform'].to(device) / wav_scale
    pred, _ = model(obj_id, force_feat, force_wav)
    loss = loss_fn(pred, target)
    if env_lambda > 0:
        loss = loss + env_lambda * _env_l1(pred, target, env_win, env_hop)
    return loss


def train_epoch_ddsp(model, loss_fn, loader, optimizer, device,
                     feat_mean, feat_std, wav_scale, id_key,
                     env_lambda=0.0, env_win=240, env_hop=120):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        loss = _ddsp_batch_loss(model, loss_fn, batch, device,
                                feat_mean, feat_std, wav_scale, id_key,
                                env_lambda, env_win, env_hop)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def eval_epoch_ddsp(model, loss_fn, loader, device,
                    feat_mean, feat_std, wav_scale, id_key):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        loss = _ddsp_batch_loss(model, loss_fn, batch, device,
                                feat_mean, feat_std, wav_scale, id_key)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def load_init_weights(model, init_from):
    """Initialise model weights from another run's best_model.pt.

    init_from is the checkpoint file or its run directory. Tensors whose
    shapes differ between checkpoint and model are skipped (e.g. the
    identity embedding when switching between material and object
    identity), and the force-curve statistics buffers are never loaded:
    they are recomputed on the current train split. Returns the list of
    skipped keys."""
    p = Path(init_from)
    if p.is_dir():
        p = p / "best_model.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]
    skip_buffers = {"force_curve_mean", "force_curve_std"}
    own = model.state_dict()
    loadable, skipped = {}, []
    for k, v in state.items():
        if k in skip_buffers or k not in own or own[k].shape != v.shape:
            skipped.append(f"{k}: ckpt{tuple(v.shape)} vs "
                           f"model{tuple(own[k].shape) if k in own else 'absent'}")
        else:
            loadable[k] = v
    model.load_state_dict(loadable, strict=False)
    epoch = ck.get("epoch")
    print(f"Initialised from {p} (epoch {epoch}); "
          f"{len(loadable)} tensors loaded, {len(skipped)} skipped")
    for s in skipped:
        print(f"  [init skip] {s}")
    return skipped


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Identity default by split: obj for within, material for heldout, whose
    # held-out objects never appear in train and would keep a random
    # per-object embedding row. --identity overrides.
    if args.identity is None:
        args.identity = "obj" if args.split == "within" else "material"
    # Best-model criterion, recorded into config and checkpoint.
    args.best_metric = "val_mrstft" if args.backbone == "ddsp" else "val_recon"

    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                          if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Save directory, keyed by ablation level, split mode and backbone
    if args.save_dir is None:
        base = PROJECT_ROOT / "experiments"
        args.save_dir = str(
            base / f"ablation_{args.ablation}_{args.split}_{args.backbone}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Split file (shared by all four ablation levels; not used with
    # --synthetic, where the corpus sizes mirror the held-out split)
    if args.split_file is None:
        args.split_file = str(SPLITS_DIR / f"splits_{args.split}.json")
    if not args.synthetic and not os.path.exists(args.split_file):
        raise FileNotFoundError(
            f"Split file not found: {args.split_file}\n"
            f"Run first:  python3 {Path(__file__).parent / 'make_splits.py'}")
    if args.env_lambda > 0 and args.backbone != "ddsp":
        raise ValueError("--env_lambda applies to the DDSP backbone only "
                         "(the CVAE loss is MSE + beta-KL on the mel)")
    if args.synthetic and args.backbone != "ddsp":
        raise ValueError("--synthetic provides waveform targets and is "
                         "supported for the DDSP backbone only")
    # Normalisation-statistics cache tag: synthetic pretraining must not
    # clobber the real-data caches under stats/
    stats_tag = f"{args.split}_synthetic" if args.synthetic else args.split

    # Descriptor dim for the chosen feature_set (recorded so eval can
    # rebuild the model). v1 = original 6-dim; pca6 = 6 PCA scores.
    args.force_feature_dim = (6 if args.feature_set in ("v1", "pca6")
                              else len(FEATURE_SETS_V2[args.feature_set]))

    # Save config
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Dataset. Real data: built over all objects so obj_to_idx is identical
    # across splits; ddsp also needs the raw onset-window waveform target;
    # pca6 needs the train-split PCA basis (fit_pca_basis.py). With
    # --synthetic: the on-the-fly pretraining corpus of synthetic_data.py
    # (sizes mirror the held-out split).
    if args.synthetic:
        from synthetic_data import (SyntheticImpactDataset,
                                    N_TRAIN, N_VAL, N_TEST)
        ir_pool = args.ir_pool or str(PROJECT_ROOT / "cache" /
                                      "synthetic_ir_pool.pt")
        if not os.path.exists(ir_pool):
            raise FileNotFoundError(
                f"IR pool not found: {ir_pool}\n"
                f"Build it first:  python3 "
                f"{Path(__file__).parent / 'make_synthetic_ir_pool.py'}")
        dataset = SyntheticImpactDataset(
            ir_pool, base_seed=args.synthetic_base_seed,
            n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST)
        train_idx = list(range(0, N_TRAIN))
        val_idx = list(range(N_TRAIN, N_TRAIN + N_VAL))
        test_idx = list(range(N_TRAIN + N_VAL, N_TRAIN + N_VAL + N_TEST))
    else:
        pca_basis = (str(Path(__file__).resolve().parent.parent / "stats" /
                         f"pca6_basis_{args.split}.pt")
                     if args.feature_set == "pca6" else None)
        dataset = ObjectFolderRealDataset(
            data_root=args.data_root,
            return_waveform=(args.backbone == "ddsp"),
            feature_set=args.feature_set,
            pca_basis=pca_basis)
        train_idx, val_idx, test_idx = load_split_indices(dataset, args.split_file)

    # Global mel z-score stats from the train split
    mel_mean, mel_std = get_mel_stats(dataset, train_idx, stats_tag)
    dataset.set_mel_stats(mel_mean, mel_std)

    # Force conditioning stats from the train split (L4 input): a scalar
    # waveform z-score for the CVAE's 1D-CNN encoder, or the per-channel
    # [RMS, max-abs] force-curve stats set on the DDSP model once it is built.
    fw_mean = fw_std = None
    wav_scale = None
    if args.backbone == "cvae":
        fw_mean, fw_std = get_force_wav_stats(dataset, train_idx, stats_tag)
        fw_mean = fw_mean.to(device)
        fw_std = fw_std.to(device)
    else:  # ddsp
        # Global waveform target scale (train-split std)
        wav_scale = get_wav_scale(dataset, train_idx, stats_tag).to(device)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    # Force descriptor normalisation stats (global z-score, train split)
    train_features = [dataset[i]['force_features'] for i in train_idx]
    feat_mean, feat_std = normalize_features(train_features)
    feat_mean = feat_mean.to(device)
    feat_std = feat_std.to(device)

    # Save normalization stats
    torch.save({'mean': feat_mean.cpu(), 'std': feat_std.cpu()},
               os.path.join(args.save_dir, "feature_stats.pt"))

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Determine mel spectrogram shape from first sample
    sample = dataset[0]
    n_mels = sample['mel'].shape[0]
    n_frames = sample['mel'].shape[1]
    force_wav_samples = sample['force_waveform'].shape[0]
    print(f"Mel shape: ({n_mels}, {n_frames})")
    print(f"Force waveform samples: {force_wav_samples}")

    # Which batch key feeds the identity embedding.
    id_key = 'obj_id' if args.identity == 'obj' else 'material_idx'

    # Model
    loss_fn = None
    if args.backbone == "ddsp":
        # Filterbank-DDSP hybrid backbone (model_ddsp.py)
        from model_ddsp import build_model, MRSTFTLoss
        model = build_model(
            n_objects=dataset.n_objects,
            ablation_level=args.ablation,
            n_bands=args.n_bands,
            use_modal=(args.modal == "on"),
            signal_samples=sample['waveform'].shape[0],
            force_waveform_samples=force_wav_samples,
            identity=args.identity,
            n_materials=dataset.n_materials,
            force_feature_dim=args.force_feature_dim,
        ).to(device)
        loss_fn = MRSTFTLoss()
        # per-channel force-curve z-score stats (train split) into the model
        fc_mean, fc_std = get_force_curve_stats(
            dataset, train_idx, stats_tag, model.force_frames, model.w)
        model.set_force_curve_stats(fc_mean.to(device), fc_std.to(device))
    else:
        model = ConditionalMelVAE(
            n_mels=n_mels,
            n_frames=n_frames,
            latent_dim=args.latent_dim,
            n_objects=dataset.n_objects,
            ablation_level=args.ablation,
            force_waveform_samples=force_wav_samples,
            identity=args.identity,
            n_materials=dataset.n_materials,
        ).to(device)

    # Optional weight initialisation from another run's checkpoint (e.g. the
    # synthetic-pretraining run for the fine-tuning variant). The
    # force-curve statistics buffers are excluded inside load_init_weights,
    # so the train-split statistics set above always win.
    if args.init_from:
        load_init_weights(model, args.init_from)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} (ablation level {args.ablation})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_metric = float('inf')   # val_recon (cvae) / val MRSTFT (ddsp)
    history = []

    print(f"\nTraining ablation level {args.ablation} "
          f"(split={args.split}, backbone={args.backbone}) "
          f"for {args.epochs} epochs...")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        if args.synthetic:
            dataset.set_epoch(epoch)   # redraw the training portion
        if args.backbone == "ddsp":
            # MRSTFT on waveform target; no beta/KL machinery
            t0 = time.time()
            train_loss = train_epoch_ddsp(
                model, loss_fn, train_loader, optimizer, device,
                feat_mean, feat_std, wav_scale, id_key,
                env_lambda=args.env_lambda, env_win=args.env_win,
                env_hop=args.env_hop)
            val_loss = eval_epoch_ddsp(
                model, loss_fn, val_loader, device,
                feat_mean, feat_std, wav_scale, id_key)
            scheduler.step()
            dt = time.time() - t0

            val_metric = val_loss

            history.append({
                'epoch': epoch,
                'train_loss': train_loss, 'val_loss': val_loss,
                'lr': scheduler.get_last_lr()[0],
                'epoch_s': round(dt, 2),
            })

            if epoch % 10 == 0 or epoch == 1:
                print(f"[{epoch:3d}/{args.epochs}] "
                      f"train={train_loss:.4f} val={val_loss:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.6f} ({dt:.1f}s)")
        else:
            beta = get_beta(epoch, args.beta_max, args.beta_cycle, args.beta_ratio)

            t0 = time.time()
            train_loss, train_recon, train_kl = train_epoch(
                model, train_loader, optimizer, device, beta, args.free_bits,
                feat_mean, feat_std, fw_mean, fw_std, id_key)

            val_loss, val_recon, val_kl = eval_epoch(
                model, val_loader, device, beta, args.free_bits,
                feat_mean, feat_std, fw_mean, fw_std, id_key)

            scheduler.step()
            dt = time.time() - t0

            # val_recon is independent of beta; val_loss = recon + beta*KL
            # would favour the low-beta epochs of the cyclical schedule.
            val_metric = val_recon

            history.append({
                'epoch': epoch,
                'train_loss': train_loss, 'train_recon': train_recon, 'train_kl': train_kl,
                'val_loss': val_loss, 'val_recon': val_recon, 'val_kl': val_kl,
                'beta': beta, 'lr': scheduler.get_last_lr()[0],
            })

            if epoch % 10 == 0 or epoch == 1:
                print(f"[{epoch:3d}/{args.epochs}] "
                      f"train={train_loss:.4f} (r={train_recon:.4f} k={train_kl:.4f}) "
                      f"val={val_loss:.4f} (r={val_recon:.4f} k={val_kl:.4f}) "
                      f"β={beta:.3f} lr={scheduler.get_last_lr()[0]:.6f} "
                      f"({dt:.1f}s)")

        # Save best
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metric': val_metric,
                'best_metric': args.best_metric,
                'config': vars(args),
            }, os.path.join(args.save_dir, "best_model.pt"))

    # Save history
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': vars(args),
    }, os.path.join(args.save_dir, "final_model.pt"))

    print(f"\nDone! Best {args.best_metric}: {best_val_metric:.4f}")
    print(f"Saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
