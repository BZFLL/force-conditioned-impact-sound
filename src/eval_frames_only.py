#!/usr/bin/env python3
"""
Post-hoc control: the frames-only cell L4' (level 5 in the code)
= identity + force frames with NO summary (no peak, no descriptor).

The paper's ladder is cumulative, so L4 - L3 measures what the frames add ON
TOP of the descriptor; it cannot say whether the descriptor is redundant once
the frames are present. L4' answers that. Reading rule:

  L4' - L4  not detected  -> the summary is redundant given the frames
  L4' - L4  > 0 (worse)   -> summary and frames are complementary
  L4' - L4  < 0 (better)  -> the summary hurts once the frames are present

"Detected" = the paper's criterion: paired bootstrap over objects, 2,000
resamples, 95 % percentile CI excluding zero, two-sided p < 0.05, |d| >= 0.2.
Comparisons in this file are a new family; Holm is applied within it
(L4'-L4, L4'-L3, L4'-L1; labelled L4prime-L4, L4prime-L3, L4prime-L1 in the result file).

Test split, same machinery as src/evaluate.py (object-first VGGish FAD,
per-object seed averaging).

Usage:
  python3 src/eval_frames_only.py --split within  --ckpt experiments
  python3 src/eval_frames_only.py --split heldout --ckpt experiments
Writes results/frames_only_<split>_test.json.
"""
import os
import sys
import json
import argparse
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as E
from dataset import ObjectFolderRealDataset, compute_mel_stats, load_split_indices

LEVELS_NEEDED = (1, 3, 4, 5)
LEVEL_NAME = {1: "L1", 3: "L3", 4: "L4", 5: "L4prime"}
FAMILY = [("L4prime-L4", 5, 4), ("L4prime-L3", 5, 3), ("L4prime-L1", 5, 1)]


def ladder_cells(roots, level, split, backbone="ddsp",
                   seeds=(42, 43, 44, 45, 46)):
    """The run directories of one ladder cell:
    <root>/ablation_<level>_<split>_<backbone>_s<seed>/best_model.pt, matched by exact
    directory name."""
    out = []
    for root in roots:
        for s in seeds:
            d = Path(root) / f"ablation_{level}_{split}_{backbone}_s{s}"
            if (d / "best_model.pt").exists() and d not in out:
                out.append(d)
    return sorted(out, key=lambda d: d.name)


def seed_avg_per_object(evals):
    acc = {}
    for ev in evals:
        for o, f in ev["fad_by_obj"].items():
            acc.setdefault(o, []).append(f)
    return {o: float(np.mean(v)) for o, v in acc.items()}


def compare(a_by_obj, b_by_obj, label):
    """delta = A - B per object (negative: A better); paired bootstrap over objects."""
    common = sorted(set(a_by_obj) & set(b_by_obj))
    delta = {o: a_by_obj[o] - b_by_obj[o] for o in common}
    mean, ci, boot = E.paired_bootstrap_objects(delta, seed=0)
    vals = np.array([delta[o] for o in common])
    return dict(comparison=label, n_objects=len(common), delta=mean, ci95=list(ci),
                cohens_d_paired=E.cohens_d(vals), p_two_sided=E.bootstrap_p_two_sided(boot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["within", "heldout"], required=True)
    ap.add_argument("--backbone", choices=["ddsp", "cvae"], default="ddsp",
                    help="which backbone's ladder to compare; DDSP is the default. "
                         "On the CVAE the force route is "
                         "L4's global 32-D code of the window, not the per-frame curve, "
                         "so the L4'-L4 reading is about a different force route")
    ap.add_argument("--ckpt", nargs="+", default=["experiments"],
                    help="root folders holding ablation_{1,3,4,5}_<split>_<backbone>_s<seed>/best_model.pt "
                         "(several roots allowed)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[frames-only] device={device} split={args.split} eval_split=test")

    dataset = ObjectFolderRealDataset(return_waveform=(args.backbone == "ddsp"))
    split_idx = load_split_indices(dataset, str(E.SPLITS_DIR / f"splits_{args.split}.json"))
    train_idx, _, _ = split_idx
    mel_mean, mel_std = compute_mel_stats(dataset, train_idx)
    dataset.set_mel_stats(mel_mean, mel_std)
    feat = torch.stack([dataset[i]["force_features"] for i in train_idx])
    feat_mean = feat.mean(0); feat_std = feat.std(0); feat_std[feat_std < 1e-6] = 1.0
    fwv = torch.stack([dataset[i]["force_waveform"] for i in train_idx])
    fw_stats = (fwv.mean(), fwv.std().clamp(min=1e-6))
    n_frames = dataset[0]["mel"].shape[1]
    gl_inv = E.GriffinLimInverter(mel_mean, mel_std, n_frames, device)
    encoder, is_fallback = E.build_fad_encoder(device)
    if is_fallback:
        raise SystemExit("[frames-only] the VGGish FAD encoder could not be loaded")

    by_level = {lvl: ladder_cells(args.ckpt, lvl, args.split, args.backbone)
                for lvl in LEVELS_NEEDED}
    for lvl in LEVELS_NEEDED:
        dirs = by_level[lvl]
        print(f"[frames-only] {LEVEL_NAME[lvl]}: {len(dirs)} checkpoints {[d.name for d in dirs]}")
        if not dirs:
            raise SystemExit(f"[frames-only] no checkpoints for {LEVEL_NAME[lvl]} under {args.ckpt}")

    per_level = {}
    for lvl in LEVELS_NEEDED:
        evals = [E.eval_one_model(d, dataset, split_idx, encoder, device, gl_inv,
                                  (feat_mean, feat_std), fw_stats, None, "test")
                 for d in sorted(by_level[lvl], key=lambda d: d.name)]
        per_level[lvl] = dict(
            n_seeds=len(evals),
            per_seed=[{"ckpt": ev["meta"]["ckpt_dir"], "seed": ev["meta"]["seed"],
                       "object_first_fad": float(np.mean(list(ev["fad_by_obj"].values())))}
                      for ev in evals],
            fad_by_obj=seed_avg_per_object(evals))
        per_level[lvl]["object_first_fad_mean"] = float(np.mean(list(per_level[lvl]["fad_by_obj"].values())))
        print(f"[frames-only] {LEVEL_NAME[lvl]} object-first FAD (seed-avg) = {per_level[lvl]['object_first_fad_mean']:.4f}")

    comps = [compare(per_level[a]["fad_by_obj"], per_level[b]["fad_by_obj"], lab)
             for lab, a, b in FAMILY]
    holm = E.holm_correction({c["comparison"]: c["p_two_sided"] for c in comps})
    for c in comps:
        c["p_holm"] = holm[c["comparison"]]
        c["detected"] = bool(c["p_holm"] < 0.05 and (c["ci95"][0] > 0 or c["ci95"][1] < 0)
                             and abs(c["cohens_d_paired"]) >= 0.2)
    l4p_l4 = comps[0]
    if not l4p_l4["detected"]:
        reading = "L4prime-L4 not detected: the summary is redundant given the frames"
    elif l4p_l4["delta"] > 0:
        reading = "L4prime worse than L4: summary and frames are complementary"
    else:
        reading = "L4prime better than L4: the summary hurts once the frames are present"

    report = {"experiment": "frames_only",
              "backbone": args.backbone,
              "split": args.split, "eval_split": "test", "metric": "object_first_VGGish_FAD",
              "device": str(device), "levels": {LEVEL_NAME[k]: {kk: vv for kk, vv in v.items() if kk != "fad_by_obj"}
                                                 for k, v in per_level.items()},
              "family_holm": comps, "reading": reading}
    default_name = (f"frames_only_{args.split}_test.json" if args.backbone == "ddsp"
                    else f"frames_only_{args.backbone}_{args.split}_test.json")
    out = Path(args.out) if args.out else E.RESULTS_DIR / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(E._jsonify(report), open(out, "w"), indent=2)
    for c in comps:
        print(f"[frames-only] {c['comparison']:10s} delta={c['delta']:+.4f} CI[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] "
              f"d={c['cohens_d_paired']:+.2f} p={c['p_two_sided']:.4f} p_holm={c['p_holm']:.4f} detected={c['detected']}")
    print(f"[frames-only] READING: {reading}")
    print(f"[frames-only] wrote {out}")


if __name__ == "__main__":
    main()
