#!/usr/bin/env python3
"""
Template-pulse control, DDSP backbone.

The template-pulse control (level key "template" in the result file) is
DDSP L4 whose per-frame force route receives the split's template pulse
scaled to the strike's own window peak (dataset.py, force_input=
"template_peak") instead of the measured 200 ms window; the descriptor still
comes from the measured force. The frame-shuffle control keeps a real pulse
from another strike; the template-pulse control removes all strike-specific shape and keeps the
interface, the timing and the peak.

Reading rule, per split, delta = FAD(template) - FAD(L4), "detected" with the
family's Holm p:
  row 1  template-L4 detected, delta > 0         -> the generator uses the measured
                                                    time course beyond timing and peak
  row 2  template-L4 not detected, template-L3   -> template-L4 written as not detected,
         detected with                              template-L3 as an improvement;
         FAD(template) - FAD(L3) < 0                informative_null is true on the seen
                                                    split when the template-L4 CI lies
                                                    inside +/-0.11 (power statement)
  row 3  neither template-L4 nor template-L3     -> both numbers
         detected
  row 4  template-L4 detected, delta < 0         -> written as it is
Any other combination is written as it is.

"Detected" = the paper's criterion: paired bootstrap over objects, 2,000
resamples, 95 % percentile CI excluding zero, two-sided p < 0.05, |d| >= 0.2.
Comparisons form the family {template-L4, template-L3, template-L1}; Holm is
applied within it. template-L4prime (L4prime = the ablation_5 frames-only
cells, level 5 in the code) is secondary: outside the family, no Holm, no
reading.

Evaluation only, test split.

Usage (PYTORCH_ENABLE_MPS_FALLBACK=1):
  python3 src/eval_template_pulse.py --split within  --ckpt experiments --tckpt experiments_template
  python3 src/eval_template_pulse.py --split heldout --ckpt experiments --tckpt experiments_template
Writes results/template_pulse_<split>_test.json (new file).
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import evaluate as E
from dataset import ObjectFolderRealDataset, compute_mel_stats, load_split_indices

SEEDS = (42, 43, 44, 45, 46)
LADDER_LEVELS = {"L1": 1, "L3": 3, "L4": 4, "L4prime": 5}
FAMILY = [("template-L4", "template", "L4"), ("template-L3", "template", "L3"),
          ("template-L1", "template", "L1")]
SECONDARY = ("template-L4prime", "template", "L4prime")
# Power statement: a seen-split template-L4 null is informative
# only if its CI lies inside +/-0.11 (the design's detectable effect).
NULL_BOUND = 0.11
SCORING_ORDER = ("L1", "L4", "L3", "L4prime", "template")


def ladder_cells(roots, level, split, backbone="ddsp", seeds=SEEDS):
    """The run directories of one ladder cell, and nothing else:
    <root>/ablation_<level>_<split>_<backbone>_s<seed>/best_model.pt. Runs are matched
    by exact directory name."""
    out = []
    for root in roots:
        for s in seeds:
            d = Path(root) / f"ablation_{level}_{split}_{backbone}_s{s}"
            if (d / "best_model.pt").exists() and d not in out:
                out.append(d)
    return sorted(out, key=lambda d: d.name)


def template_cells(root, split, seeds=SEEDS):
    """<root>/template_4_<split>_ddsp_s<seed>/best_model.pt."""
    out = []
    for s in seeds:
        d = Path(root) / f"template_4_{split}_ddsp_s{s}"
        if (d / "best_model.pt").exists():
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


def detected(c, p_key):
    return bool(c[p_key] < 0.05 and (c["ci95"][0] > 0 or c["ci95"][1] < 0)
                and abs(c["cohens_d_paired"]) >= 0.2)


def reading_row(t_l4, t_l3, split):
    """The reading-rule row for this split, with the sentence filled in."""
    where = "seen" if split == "within" else "unseen"
    num = lambda c: f"{c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
    if t_l4["detected"] and t_l4["delta"] > 0:
        return dict(row=1, sentence=(
            f"Replacing the measured frames by a template pulse scaled to the strike's peak "
            f"costs {num(t_l4)} on {where} objects, so the generator uses the measured time "
            f"course beyond its timing and peak."))
    if t_l4["detected"] and t_l4["delta"] < 0:
        return dict(row=4, sentence=f"template-L4 detected with delta < 0 (template better) on {where} "
                                    f"objects: {num(t_l4)}; written as it is.")
    if t_l3["detected"] and t_l3["delta"] < 0:
        informative_null = bool(split == "within" and t_l4["ci95"][0] >= -NULL_BOUND
                                and t_l4["ci95"][1] <= NULL_BOUND)
        return dict(row=2, informative_null=informative_null, sentence=(
            f"Replacing the measured frames by a template pulse scaled to the strike's peak is not "
            f"detected on {where} objects ({num(t_l4)}), while the template still improves on the "
            f"descriptor ({num(t_l3)})."))
    if not t_l3["detected"]:
        return dict(row=3, sentence=(f"On {where} objects neither template-L4 ({num(t_l4)}) nor "
                                     f"template-L3 "
                                     f"({num(t_l3)}) is detected."))
    return dict(row=None, sentence=(f"On {where} objects template-L4 is not detected ({num(t_l4)}) and "
                                    f"template-L3 is detected with FAD(template) - FAD(L3) > 0 "
                                    f"({num(t_l3)})."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["within", "heldout"], required=True)
    ap.add_argument("--ckpt", nargs="+", default=["experiments"],
                    help="root folders holding ablation_{1,3,4,5}_<split>_ddsp_s<seed>/best_model.pt "
                         "(several roots allowed)")
    ap.add_argument("--tckpt", default="experiments_template",
                    help="root folder holding template_4_<split>_ddsp_s<seed>/best_model.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t_start = time.time()

    out = Path(args.out) if args.out else E.RESULTS_DIR / f"template_pulse_{args.split}_test.json"

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[template] device={device} split={args.split} (test split, evaluation only)")

    template_path = ROOT / "stats" / f"template_pulse_{args.split}.pt"
    cache_path = ROOT / "stats" / f"force_curve_stats_{args.split}.pt"
    for p in (template_path, cache_path):
        if not p.exists():
            raise SystemExit(f"[template] missing {p}")
    tmpl = torch.load(template_path, map_location="cpu", weights_only=False)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"[template] template {template_path.name}")
    print(f"[template] force-curve stats {cache_path.name} "
          f"mean={cache['mean'].tolist()} std={cache['std'].tolist()}")

    # Checkpoints.
    by_level = {name: ladder_cells(args.ckpt, lvl, args.split)
                for name, lvl in LADDER_LEVELS.items()}
    by_level["template"] = template_cells(args.tckpt, args.split)
    for name in SCORING_ORDER:
        dirs = by_level[name]
        print(f"[template] {name}: {len(dirs)} checkpoints {[d.name for d in dirs]}")

    # Measured dataset (L1, L3, L4, L4prime and every statistic) and template
    # dataset (template level only).
    dataset = ObjectFolderRealDataset(return_waveform=True)
    split_file = str(E.SPLITS_DIR / f"splits_{args.split}.json")
    split_idx = load_split_indices(dataset, split_file)
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
        raise SystemExit("[template] FAD encoder fell back; VGGish is required")

    dataset_t = ObjectFolderRealDataset(return_waveform=True, force_input="template_peak",
                                        template_path=str(template_path))
    dataset_t.set_mel_stats(mel_mean, mel_std)

    per_level = {}
    for name in SCORING_ORDER:
        ds = dataset_t if name == "template" else dataset
        t0 = time.time()
        evals = [E.eval_one_model(d, ds, split_idx, encoder, device, gl_inv,
                                  (feat_mean, feat_std), fw_stats, None, "test")
                 for d in by_level[name]]
        per_level[name] = dict(
            n_seeds=len(evals),
            per_seed=[{"ckpt": ev["meta"]["ckpt_dir"], "seed": ev["meta"]["seed"],
                       "object_first_fad": float(np.mean(list(ev["fad_by_obj"].values())))}
                      for ev in evals],
            fad_by_obj=seed_avg_per_object(evals),
            scoring_seconds=round(time.time() - t0, 1))
        per_level[name]["object_first_fad_mean"] = float(np.mean(list(per_level[name]["fad_by_obj"].values())))
        print(f"[template] {name}: {len(evals)} checkpoints, object-first FAD (seed-avg) = "
              f"{per_level[name]['object_first_fad_mean']:.4f}  ({per_level[name]['scoring_seconds']:.0f}s)")

    comps = [compare(per_level[a]["fad_by_obj"], per_level[b]["fad_by_obj"], lab)
             for lab, a, b in FAMILY]
    holm = E.holm_correction({c["comparison"]: c["p_two_sided"] for c in comps})
    for c in comps:
        c["p_holm"] = holm[c["comparison"]]
        c["detected"] = detected(c, "p_holm")
    secondary = compare(per_level[SECONDARY[1]]["fad_by_obj"], per_level[SECONDARY[2]]["fad_by_obj"], SECONDARY[0])
    secondary["detected_uncorrected"] = detected(secondary, "p_two_sided")
    by_name = {c["comparison"]: c for c in comps}
    reading = reading_row(by_name["template-L4"], by_name["template-L3"], args.split)

    report = {"experiment": "template_pulse",
              "backbone": "ddsp", "split": args.split, "eval_split": "test",
              "metric": "object_first_VGGish_FAD", "device": str(device),
              "template": {"file": f"stats/{template_path.name}",
                           "n_train": tmpl.get("n_train"), "peak_index": tmpl.get("peak_index"),
                           "n_peak_mismatch": tmpl.get("n_peak_mismatch")},
              "force_curve_stats": {"file": f"stats/{cache_path.name}",
                                    "mean": cache["mean"].tolist(), "std": cache["std"].tolist()},
              "levels": {k: {kk: vv for kk, vv in v.items() if kk != "fad_by_obj"} for k, v in per_level.items()},
              "family_holm": comps, "secondary_no_holm": secondary,
              "reading": reading, "scoring_wall_seconds": round(time.time() - t_start, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(E._jsonify(report), open(out, "w"), indent=2)
    for c in comps + [secondary]:
        p_part = f" p_holm={c['p_holm']:.4f} detected={c['detected']}" if "p_holm" in c else \
                 f" (no Holm) detected_uncorrected={c['detected_uncorrected']}"
        print(f"[template] {c['comparison']:16s} delta={c['delta']:+.4f} CI[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] "
              f"d={c['cohens_d_paired']:+.3f} p={c['p_two_sided']:.4f}" + p_part)
    print(f"[template] READING row {reading['row']}: {reading['sentence']}")
    print(f"[template] wrote {out} (wall {report['scoring_wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
