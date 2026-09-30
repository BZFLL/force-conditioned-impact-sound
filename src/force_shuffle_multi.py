#!/usr/bin/env python3
"""
Frame-shuffle control over SEVERAL reassignments.

The single-reassignment result in the paper's Table 2 ("L4, frame shuffle - L4")
uses one fixed reassignment (numpy default_rng(42)). One draw cannot show whether
the reported effect depends on that draw. This script repeats the control for
N reassignments (seeds 42, 43, ...); reassignment seed 42 is the single
reassignment reported in Table 2.

For each L4 DDSP checkpoint (5 seeds) and each reassignment, every test strike
is generated twice: with its own force frames, and with another test strike's
frames (identity and descriptor stay correct). Object-first VGGish FAD of both
against the same real audio; delta = FAD(shuffled) - FAD(correct); positive
means the wrong frames hurt. Per reassignment: seed-averaged per-object delta,
paired bootstrap over objects (2,000 resamples). Across reassignments: mean,
s.d. and range of the delta; the paired test is NOT pooled across
reassignments (they are not independent observations of new objects).

Usage:
  python3 src/force_shuffle_multi.py --split within  --n-reassignments 10
  python3 src/force_shuffle_multi.py --split heldout --n-reassignments 10
Writes results/force_shuffle_multi_<split>_test.json (new file).
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


def ladder_cells(roots, level, split, seeds=(42, 43, 44, 45, 46)):
    """The run directories of one ladder cell, and nothing else:
    <root>/ablation_<level>_<split>_ddsp_s<seed>/best_model.pt. Runs are matched
    by exact directory name."""
    out = []
    for root in roots:
        for s in seeds:
            d = Path(root) / f"ablation_{level}_{split}_ddsp_s{s}"
            if (d / "best_model.pt").exists() and d not in out:
                out.append(d)
    return sorted(out, key=lambda d: d.name)


def reassignment(n, rng):
    """Random permutation with no fixed points (same construction as the single-reassignment control in Table 2)."""
    p = rng.permutation(n)
    for i in range(n):
        if p[i] == i:
            j = (i + 1) % n
            p[i], p[j] = p[j], p[i]
    for i in range(n):
        if p[i] == i:
            p[i], p[(i + 1) % n] = p[(i + 1) % n], p[i]
    return p


@torch.no_grad()
def generate_correct(model, dataset, idx, device, feat_stats, id_key):
    feat_mean, feat_std = feat_stats
    out = {}
    for i in idx:
        s = dataset[i]
        ident = torch.tensor([s[id_key]], dtype=torch.long, device=device)
        ff = ((s["force_features"] - feat_mean) / feat_std).unsqueeze(0).to(device)
        lo = torch.zeros(1, dtype=torch.long, device=device)
        fw = s["force_waveform"].unsqueeze(0).to(device)
        out[i] = dict(ident=ident, ff=ff, lo=lo, fw=fw,
                      gen_ok=model(ident, ff, fw, loop_offsets=lo)[0][0].cpu(),
                      gt=s["waveform"].cpu(), obj=s["obj_id_raw"])
    return out


@torch.no_grad()
def generate_shuffled(model, cache, idx, perm):
    out = {}
    for pos, i in enumerate(idx):
        c = cache[i]
        fw_other = cache[idx[perm[pos]]]["fw"]
        out[i] = model(c["ident"], c["ff"], fw_other, loop_offsets=c["lo"])[0][0].cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["within", "heldout"], required=True)
    ap.add_argument("--n-reassignments", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=42,
                    help="reassignment seeds run from here; 42 is the single reassignment reported in Table 2")
    ap.add_argument("--ckpt", nargs="+", default=["experiments"],
                    help="root folders holding ablation_4_<split>_ddsp_s<seed>/best_model.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[shuffle-multi] device={device} split={args.split} test split, "
          f"{args.n_reassignments} reassignments from seed {args.first_seed}")

    dataset = ObjectFolderRealDataset(return_waveform=True)
    split_idx = load_split_indices(dataset, str(E.SPLITS_DIR / f"splits_{args.split}.json"))
    train_idx, _, test_idx = split_idx
    mel_mean, mel_std = compute_mel_stats(dataset, train_idx)
    dataset.set_mel_stats(mel_mean, mel_std)
    feat = torch.stack([dataset[i]["force_features"] for i in train_idx])
    feat_mean = feat.mean(0); feat_std = feat.std(0); feat_std[feat_std < 1e-6] = 1.0
    encoder, is_fallback = E.build_fad_encoder(device)
    if is_fallback:
        raise SystemExit("[shuffle-multi] FAD encoder fell back; VGGish is required")

    l4_dirs = ladder_cells(args.ckpt, 4, args.split)
    print(f"[shuffle-multi] L4 checkpoints: {[d.name for d in l4_dirs]}")

    test = sorted(test_idx)
    seeds = list(range(args.first_seed, args.first_seed + args.n_reassignments))
    perms = {s: reassignment(len(test), np.random.default_rng(s)) for s in seeds}

    # correct generations and their embeddings once per checkpoint
    per_ckpt = []
    for d in l4_dirs:
        model, meta = E.load_checkpoint_and_model(d, dataset, device)
        if not meta["force_curve_from_checkpoint"]:
            fw = torch.stack([dataset[i]["force_waveform"] for i in train_idx])
            flat = model.pool_force_curve(fw, model.force_frames, model.w).reshape(-1, 2)
            model.set_force_curve_stats(flat.mean(0).to(device), flat.std(0).clamp(min=1e-8).to(device))
        model.eval()
        id_key = "obj_id" if meta["identity"] == "obj" else "material_idx"
        cache = generate_correct(model, dataset, test, device, (feat_mean, feat_std), id_key)
        objs = [cache[i]["obj"] for i in test]
        gt_embs = np.stack([encoder.embed(cache[i]["gt"]) for i in test])
        ok_embs = np.stack([encoder.embed(cache[i]["gen_ok"]) for i in test])
        fad_ok = E.per_object_fad(ok_embs, gt_embs, objs)
        per_ckpt.append(dict(name=d.name, model=model, cache=cache, objs=objs,
                             gt_embs=gt_embs, fad_ok=fad_ok))
        print(f"[shuffle-multi] {d.name}: correct FAD = {np.mean(list(fad_ok.values())):.4f}")

    results = []
    for s in seeds:
        ok_acc, shuf_acc = {}, {}
        per_seed = []
        for pc in per_ckpt:
            gen_shuf = generate_shuffled(pc["model"], pc["cache"], test, perms[s])
            shuf_embs = np.stack([encoder.embed(gen_shuf[i]) for i in test])
            fad_shuf = E.per_object_fad(shuf_embs, pc["gt_embs"], pc["objs"])
            for o in pc["fad_ok"]:
                ok_acc.setdefault(o, []).append(pc["fad_ok"][o])
                shuf_acc.setdefault(o, []).append(fad_shuf[o])
            per_seed.append(dict(ckpt=pc["name"],
                                 fad_correct=float(np.mean(list(pc["fad_ok"].values()))),
                                 fad_shuffled=float(np.mean(list(fad_shuf.values())))))
        common = sorted(set(ok_acc) & set(shuf_acc))
        delta = {o: float(np.mean(shuf_acc[o]) - np.mean(ok_acc[o])) for o in common}
        mean, ci, boot = E.paired_bootstrap_objects(delta, seed=7)
        vals = np.array([delta[o] for o in common])
        r = dict(reassignment_seed=s, n_objects=len(common), per_seed=per_seed,
                 delta_shuf_minus_ok=mean, ci95=list(ci), cohens_d_paired=E.cohens_d(vals),
                 p_two_sided=E.bootstrap_p_two_sided(boot),
                 frac_objects_worse=float(np.mean(vals > 0)))
        results.append(r)
        print(f"[shuffle-multi] reassignment {s}: delta={mean:+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"d={r['cohens_d_paired']:+.3f} p={r['p_two_sided']:.4f}")

    deltas = np.array([r["delta_shuf_minus_ok"] for r in results])
    summary = dict(n_reassignments=len(results), delta_mean=float(deltas.mean()),
                   delta_sd=float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                   delta_min=float(deltas.min()), delta_max=float(deltas.max()),
                   n_detected=int(sum(1 for r in results
                                      if r["p_two_sided"] < 0.05 and r["ci95"][0] > 0
                                      and abs(r["cohens_d_paired"]) >= 0.2)))
    report = dict(experiment="frame_shuffle", split=args.split,
                  eval_split="test", metric="object_first_VGGish_FAD", device=str(device),
                  n_test_clips=len(test), reassignment_seeds=seeds,
                  table2_reassignment_seed=42, per_reassignment=results,
                  across_reassignments=summary)
    out = Path(args.out) if args.out else E.RESULTS_DIR / f"force_shuffle_multi_{args.split}_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(E._jsonify(report), open(out, "w"), indent=2)
    print(f"[shuffle-multi] across {len(results)} reassignments: delta mean {summary['delta_mean']:+.4f} "
          f"sd {summary['delta_sd']:.4f} range [{summary['delta_min']:+.4f}, {summary['delta_max']:+.4f}] "
          f"detected in {summary['n_detected']}/{len(results)}")
    print(f"[shuffle-multi] wrote {out}")


if __name__ == "__main__":
    main()
