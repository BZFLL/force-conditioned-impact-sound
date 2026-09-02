#!/usr/bin/env python3
"""Fair-descriptor ladder evaluation.

Evaluates the nested conditioning ladder
  L1 -> {L2 (peak scalar), L2' (impulse)} -> {L3 (6-D), L3'_A, L3'_AB} -> L4
(optionally plus L3_pca6) on one split/backbone, reusing the evaluate.py
machinery (object-level VGGish FAD, paired bootstrap over objects, Cohen's
d). Each condition's dataset is built with its own feature set so that the
descriptor content matches training.

--dev evaluates on the validation split; otherwise the test split is used.
The four main-grid levels (L1/L2/L3/L4) are evaluated from their
checkpoints in the same run. Holm
correction is applied to the vs-L1 family; the
rung comparisons (L3'_A vs L3, L3'_AB vs L3'_A, L2' vs L2, L4 vs L3'_AB)
are reported with per-comparison p. With --pca6 the L3_pca6 condition is
added together with its own three-comparison Holm family.

Usage:
    python3 src/evaluate_ladder.py --backbone ddsp --split within [--dev] [--pca6]
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as E
from dataset import ObjectFolderRealDataset, compute_mel_stats, load_split_indices


def conditions(split, backbone, include_pca6=False):
    p = f"_{split}_{backbone}"
    conds = [  # (name, feature_set, dir-prefix under experiments/, then "s{seed}")
        ("L1",    "v1",      f"ablation_1{p}_s"),
        ("L2a",   "v1",      f"ablation_2{p}_s"),
        ("L2p",   "impulse", f"ablation_2{p}_impulse_s"),
        ("L3a",   "v1",      f"ablation_3{p}_s"),
        ("L3pA",  "v2A",     f"ablation_3{p}_v2A_s"),
        ("L3pAB", "v2AB",    f"ablation_3{p}_v2AB_s"),
        ("L4",    "v1",      f"ablation_4{p}_s"),
    ]
    if include_pca6:
        conds.insert(6, ("L3pca", "pca6", f"ablation_3{p}_pca6_s"))
    return conds

VS_L1 = [("L2a", "L1"), ("L2p", "L1"), ("L3a", "L1"),
         ("L3pA", "L1"), ("L3pAB", "L1"), ("L4", "L1")]   # Holm family
RUNGS = [("L3pA", "L3a"),    # L3'_A vs L3: re-extracted descriptors vs original
         ("L3pAB", "L3pA"),  # L3'_AB vs L3'_A: the three added dimensions
         ("L2p", "L2a"),     # L2' vs L2: impulse vs peak scalar
         ("L4", "L3pAB")]    # L4 vs L3'_AB: force frames vs best descriptor
# pca6 family (own Holm correction, separate from the vs-L1 family):
PCA6_FAMILY = [("L3pca", "L1"),      # PCA-6 vs identity
               ("L3pca", "L3pAB"),   # PCA-6 vs best hand-crafted 6-D
               ("L4", "L3pca")]      # force frames vs PCA-6


def compare(cond_fad, b, a, n_boot, seed):
    common = sorted(set(cond_fad[a]) & set(cond_fad[b]))
    delta = {o: cond_fad[b][o] - cond_fad[a][o] for o in common}
    mean, ci, boot = E.paired_bootstrap_objects(delta, n_boot=n_boot, seed=seed)
    vals = np.array([delta[o] for o in common])
    return dict(delta=mean, ci95=ci, cohens_d=E.cohens_d(vals),
                p=E.bootstrap_p_two_sided(boot), n_obj=len(common))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="ddsp", choices=["ddsp", "cvae"])
    ap.add_argument("--split", default="within", choices=["within", "heldout"])
    ap.add_argument("--dev", action="store_true",
                    help="evaluate on the validation split instead of test")
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pca6", action="store_true",
                    help="include the L3_pca6 condition and its own Holm family")
    args = ap.parse_args()
    E.N_BOOTSTRAP = args.n_bootstrap
    eval_split = "val" if args.dev else "test"
    seeds = args.seeds.split(",")
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[ladder] device={device} {args.backbone}/{args.split} eval_split={eval_split}")
    if not args.dev:
        print("[ladder] evaluating on the test split")

    encoder, is_fb = E.build_fad_encoder(device)
    if is_fb:
        raise SystemExit("FAD encoder fell back to log-mel; VGGish is required.")

    cache, gl = {}, {}
    def get(fs):
        if fs not in cache:
            pb = (str(E.SPLITS_DIR.parent / "stats" / f"pca6_basis_{args.split}.pt")
                  if fs == "pca6" else None)
            ds = ObjectFolderRealDataset(return_waveform=(args.backbone == "ddsp"),
                                         feature_set=fs, pca_basis=pb)
            split_idx = load_split_indices(ds, str(E.SPLITS_DIR / f"splits_{args.split}.json"))
            tr = split_idx[0]
            mm, ms = compute_mel_stats(ds, tr); ds.set_mel_stats(mm, ms)
            feat = torch.stack([ds[i]["force_features"] for i in tr])
            fmean = feat.mean(0); fstd = feat.std(0); fstd[fstd < 1e-6] = 1.0
            fwv = torch.stack([ds[i]["force_waveform"] for i in tr])
            fw_stats = (fwv.mean(), fwv.std().clamp(min=1e-6))
            if "gl" not in gl:
                gl["gl"] = E.GriffinLimInverter(mm, ms, ds[0]["mel"].shape[1], device)
            cache[fs] = (ds, split_idx, (fmean, fstd), fw_stats)
        return cache[fs]

    cond_fad, head = {}, {}
    for name, fs, prefix in conditions(args.split, args.backbone, include_pca6=args.pca6):
        ds, split_idx, feat_stats, fw_stats = get(fs)
        dirs = [E.EXP_DIR / f"{prefix}{s}" for s in seeds]
        dirs = [d for d in dirs if (d / "best_model.pt").exists()]
        if not dirs:
            print(f"  [skip] {name}: no checkpoints ({prefix}*)"); continue
        acc = {}
        for d in dirs:
            ev = E.eval_one_model(d, ds, split_idx, encoder, device, gl["gl"],
                                  feat_stats, fw_stats, None, eval_split)
            for o, f in ev["fad_by_obj"].items():
                acc.setdefault(o, []).append(f)
        cond_fad[name] = {o: float(np.mean(v)) for o, v in acc.items()}
        head[name] = float(np.mean(list(cond_fad[name].values())))
        print(f"  {name:6s} fs={fs:7s} {len(dirs)}seed  object-first FAD = {head[name]:.4f}")

    results = {"headline_object_first_fad": head, "vs_L1": {}, "rungs": {},
               "meta": {"eval_split": eval_split, "backbone": args.backbone,
                        "split": args.split, "n_bootstrap": args.n_bootstrap,
                        "note": "delta = B - A; negative = B better (FAD lower)."}}

    # vs-L1 family with Holm
    raw_p = {}
    for i, (b, a) in enumerate(VS_L1):
        if b in cond_fad and a in cond_fad:
            c = compare(cond_fad, b, a, args.n_bootstrap, 1000 + i)
            results["vs_L1"][f"{b}-{a}"] = c; raw_p[f"{b}-{a}"] = c["p"]
    holm = E.holm_correction(raw_p) if raw_p else {}
    for k, c in results["vs_L1"].items():
        c["p_holm"] = holm.get(k)
        lo, hi = c["ci95"]
        c["verdict"] = ("B better (Holm)" if c["delta"] < 0 and hi < 0 and abs(c["cohens_d"]) >= 0.2 and holm.get(k, 1) < 0.05
                        else "B WORSE (Holm)" if c["delta"] > 0 and lo > 0 and abs(c["cohens_d"]) >= 0.2 and holm.get(k, 1) < 0.05
                        else "no detected diff")

    # rung comparisons (per-comparison p, no family correction)
    for i, (b, a) in enumerate(RUNGS):
        if b in cond_fad and a in cond_fad:
            c = compare(cond_fad, b, a, args.n_bootstrap, 2000 + i)
            lo, hi = c["ci95"]
            c["verdict"] = ("B better" if c["delta"] < 0 and hi < 0 and abs(c["cohens_d"]) >= 0.2 and c["p"] < 0.05
                            else "B WORSE" if c["delta"] > 0 and lo > 0 and abs(c["cohens_d"]) >= 0.2 and c["p"] < 0.05
                            else "no detected diff")
            results["rungs"][f"{b}-{a}"] = c

    # pca6 family: own Holm over its 3 comparisons.
    if args.pca6 and "L3pca" in cond_fad:
        results["pca6_family"] = {}
        praw = {}
        for i, (b, a) in enumerate(PCA6_FAMILY):
            if b in cond_fad and a in cond_fad:
                c = compare(cond_fad, b, a, args.n_bootstrap, 3000 + i)
                results["pca6_family"][f"{b}-{a}"] = c
                praw[f"{b}-{a}"] = c["p"]
        holmB = E.holm_correction(praw) if praw else {}
        for k, c in results["pca6_family"].items():
            c["p_holm"] = holmB.get(k)
            lo, hi = c["ci95"]
            c["verdict"] = ("B better (Holm)" if c["delta"] < 0 and hi < 0 and abs(c["cohens_d"]) >= 0.2 and holmB.get(k, 1) < 0.05
                            else "B WORSE (Holm)" if c["delta"] > 0 and lo > 0 and abs(c["cohens_d"]) >= 0.2 and holmB.get(k, 1) < 0.05
                            else "no detected diff")

    # print
    def row(k, c):
        return (f"  {k:12s} Δ={c['delta']:+.4f} CI[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] "
                f"d={c['cohens_d']:+.3f} p={c['p']:.4f}"
                + (f" pHolm={c['p_holm']:.4f}" if c.get('p_holm') is not None else "")
                + f"  → {c['verdict']}")
    print("\nvs L1 (Holm family):")
    for k, c in results["vs_L1"].items():
        print(row(k, c))
    print("\nladder rungs (per-comparison p):")
    for k, c in results["rungs"].items():
        print(row(k, c))
    if results.get("pca6_family"):
        print("\npca6 family (own Holm):")
        for k, c in results["pca6_family"].items():
            print(row(k, c))

    out = Path(args.out) if args.out else (E.RESULTS_DIR / f"ladder_{args.backbone}_{args.split}_{eval_split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(E._jsonify(results), open(out, "w"), indent=2)
    print(f"\n[ladder] wrote {out}")


if __name__ == "__main__":
    main()
