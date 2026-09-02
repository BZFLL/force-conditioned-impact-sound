#!/usr/bin/env python3
"""Validation-split object-level FAD for the reference systems of Table 2.

Scores trained DDSP checkpoints on the validation split with the same
machinery as evaluate.py (VGGish encoder, per-object Fréchet distance
averaged over objects) and pairs each variant with the from-scratch L4
checkpoint of the same seed. Used for the training-side comparisons, which
are reported on the validation split only (from-scratch L4, envelope-L1 loss,
synthetic-IR pretraining + fine-tuning).

Default run sets (three seeds each):
    within : ddsp_L4_s{42,43,44}       experiments/ablation_4_within_ddsp_s<seed>
             pretrain_ft_L4_s{42,43,44} experiments/pretrain/ft_within_L4_s<seed>
    heldout: ddsp_L4_s{42,43,44}       experiments/ablation_4_heldout_ddsp_s<seed>
             envloss_L4_s{42,43,44}     experiments/envloss/envloss_heldout_L4_s<seed>
             pretrain_ft_L4_s{42,43,44} experiments/pretrain/ft_heldout_L4_s<seed>

Usage:
    python3 src/eval_val_fad.py --split within
    python3 src/eval_val_fad.py --split heldout
    python3 src/eval_val_fad.py --split heldout --run name=path/to/run_dir [...]

Output: results/val_fad_<split>.json with per-model object-level FAD and
per-object values, plus same-seed paired comparisons against ddsp_L4.
An incremental cache (results/val_fad_<split>_partial.json) allows an
interrupted run to resume; it is removed on success.
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (build_fad_encoder, GriffinLimInverter, eval_one_model,
                      SPLITS_DIR, EXP_DIR, RESULTS_DIR)
from dataset import (ObjectFolderRealDataset, compute_mel_stats,
                     load_split_indices)

EVAL_SPLIT = "dev"   # evaluate.py maps "dev" to the validation indices
SEEDS = (42, 43, 44)
BASELINE = "ddsp_L4"


def default_runs(split):
    runs = []
    for s in SEEDS:
        runs.append((f"ddsp_L4_s{s}", EXP_DIR / f"ablation_4_{split}_ddsp_s{s}"))
    if split == "heldout":
        for s in SEEDS:
            runs.append((f"envloss_L4_s{s}",
                         EXP_DIR / "envloss" / f"envloss_heldout_L4_s{s}"))
    for s in SEEDS:
        runs.append((f"pretrain_ft_L4_s{s}",
                     EXP_DIR / "pretrain" / f"ft_{split}_L4_s{s}"))
    return runs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split", required=True, choices=["within", "heldout"])
    p.add_argument("--run", action="append", default=[],
                   help="name=run_dir; repeatable. Overrides the default set. "
                        "Names ending in _s<seed> are paired by seed with "
                        f"{BASELINE}_s<seed>.")
    p.add_argument("--out", default=None,
                   help="output JSON (default results/val_fad_<split>.json)")
    return p.parse_args()


def obj_mean(fad_by_obj):
    return float(np.mean(list(fad_by_obj.values())))


def split_name_seed(name):
    if "_s" in name and name.rsplit("_s", 1)[1].isdigit():
        base, seed = name.rsplit("_s", 1)
        return base, seed
    return name, None


def main():
    args = parse_args()
    split = args.split
    if args.run:
        runs = []
        for spec in args.run:
            name, path = spec.split("=", 1)
            runs.append((name, Path(path)))
    else:
        runs = default_runs(split)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"val_fad_{split}.json"
    partial_path = out_path.with_name(out_path.stem + "_partial.json")

    missing = [str(d) for _, d in runs if not (d / "best_model.pt").exists()]
    if missing:
        sys.exit("missing checkpoints:\n  " + "\n  ".join(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                          if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    print(f"split={split}  eval split=val  device={device}  runs={len(runs)}")

    encoder, is_fallback = build_fad_encoder(device)
    if is_fallback:
        sys.exit("VGGish could not be loaded; the log-mel fallback encoder is "
                 "not comparable with the reported numbers.")

    partial = json.load(open(partial_path)) if partial_path.exists() else {}
    if partial:
        print(f"resuming: {len(partial)} finished evaluation(s) reused")

    dataset = ObjectFolderRealDataset(return_waveform=True)
    split_idx = load_split_indices(dataset, str(SPLITS_DIR / f"splits_{split}.json"))
    train_idx, val_idx, _ = split_idx
    mel_mean, mel_std = compute_mel_stats(dataset, train_idx)
    dataset.set_mel_stats(mel_mean, mel_std)
    feat = torch.stack([dataset[i]["force_features"] for i in train_idx])
    feat_mean, feat_std = feat.mean(0), feat.std(0)
    feat_std[feat_std < 1e-6] = 1.0
    fwv = torch.stack([dataset[i]["force_waveform"] for i in train_idx])
    fw_stats = (fwv.mean(), fwv.std().clamp(min=1e-6))
    n_frames = dataset[0]["mel"].shape[1]
    gl_inv = GriffinLimInverter(mel_mean, mel_std, n_frames, device)
    print(f"n_val_clips={len(val_idx)}")

    models = {}
    for name, ckpt in runs:
        if name in partial:
            models[name] = partial[name]
            print(f"  {name}: cached  FAD={partial[name]['object_first_fad_mean']:.4f}")
            continue
        tm = time.time()
        ev = eval_one_model(ckpt, dataset, split_idx, encoder, device, gl_inv,
                            (feat_mean, feat_std), fw_stats, None, EVAL_SPLIT)
        fad_by_obj = {str(k): float(v) for k, v in ev["fad_by_obj"].items()}
        models[name] = dict(checkpoint=str(ckpt.relative_to(EXP_DIR.parent))
                            if ckpt.is_relative_to(EXP_DIR.parent) else str(ckpt),
                            object_first_fad_mean=obj_mean(fad_by_obj),
                            n_objects=len(fad_by_obj),
                            fad_by_obj=fad_by_obj)
        print(f"  {name}: FAD={models[name]['object_first_fad_mean']:.4f} "
              f"(n_obj={len(fad_by_obj)}, {time.time()-tm:.0f}s)", flush=True)
        partial[name] = models[name]
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(partial, open(partial_path, "w"))
        if device.type == "mps":
            torch.mps.empty_cache()

    # Same-seed paired comparisons against the from-scratch baseline.
    comparisons = {}
    groups = {}
    for name in models:
        base, seed = split_name_seed(name)
        groups.setdefault(base, {})[seed] = name
    for base, by_seed in groups.items():
        if base == BASELINE or BASELINE not in groups:
            continue
        per_seed = {}
        for seed, name in sorted(by_seed.items()):
            bname = groups[BASELINE].get(seed)
            if bname is None:
                continue
            m, b = models[name], models[bname]
            objs = sorted(set(m["fad_by_obj"]) & set(b["fad_by_obj"]))
            deltas = [m["fad_by_obj"][o] - b["fad_by_obj"][o] for o in objs]
            per_seed[seed] = dict(
                model=m["object_first_fad_mean"],
                baseline=b["object_first_fad_mean"],
                delta=m["object_first_fad_mean"] - b["object_first_fad_mean"],
                pct=100.0 * (m["object_first_fad_mean"] / b["object_first_fad_mean"] - 1.0),
                paired_mean_delta=float(np.mean(deltas)),
                n_objects_paired=len(objs),
                n_objects_improved=int(sum(d < 0 for d in deltas)),
                n_objects_worsened=int(sum(d > 0 for d in deltas)))
        if per_seed:
            comparisons[base] = dict(
                baseline=BASELINE, per_seed=per_seed,
                mean_model=float(np.mean([v["model"] for v in per_seed.values()])),
                mean_baseline=float(np.mean([v["baseline"] for v in per_seed.values()])),
                improves_in_all_seeds=all(v["delta"] < 0 for v in per_seed.values()))

    out = dict(split=split, eval_split="val", fad_encoder=encoder.name,
               aggregation="object_first_perobject_fad",
               models=models, comparisons=comparisons)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    if partial_path.exists():
        partial_path.unlink()

    print(f"\n{'system':28s} {'mean FAD':>9s}  per-seed")
    for base, by_seed in groups.items():
        vals = [models[n]["object_first_fad_mean"] for _, n in sorted(by_seed.items())]
        print(f"{base:28s} {np.mean(vals):9.4f}  " + "  ".join(f"{v:.4f}" for v in vals))
    print(f"wrote {out_path}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
