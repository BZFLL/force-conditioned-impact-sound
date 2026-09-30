#!/usr/bin/env python3
"""
Frame-shuffle control over SEVERAL reassignments, mel-CVAE L4.

The frame-shuffle control of force_shuffle_multi.py for the CVAE backbone.
reassignment(), the bootstrap, the per-reassignment fields and the report format
are the same as in force_shuffle_multi.py. Differences for the CVAE backbone:

- checkpoints ablation_4_<split>_cvae_s<seed> (backbone cvae, level 4); the
  DDSP force-curve-stats block is dropped;
- the dataset (return_waveform=False), fw_stats (one train-split scalar for the
  force window) and gl_inv are built as evaluate.main does for the CVAE;
- generation mirrors evaluate.generate_waveforms, CVAE branch: for every strike
  generate -> Griffin-Lim(generated mel) -> Griffin-Lim(real mel); the ground
  truth is the Griffin-Lim reconstruction of the real mel;
- one loop run_pass(model, perm) serves the correct pass (identity perm) and
  every shuffled pass, primed by one Griffin-Lim call. GriffinLimInverter
  reseeds torch (torch.manual_seed, all devices) on every call, so every strike
  in every pass receives the same latent z and a strike's correct and shuffled
  generations differ only in the force window;
- the summary adds n_detected_negative; n_detected has the same definition as
  in force_shuffle_multi.py.

On this backbone the force window reaches the model only as one global 32-D
code (ForceWaveformEncoder: three stride-2 convolutions, receptive field 23
samples, averaged over 1,200 positions). The control tests whether that code
carries strike-specific information beyond the strike's own descriptor, not
the use of a frame-level time course.

Usage:
  python3 src/force_shuffle_multi_cvae.py --split within  --n-reassignments 10 --ckpt experiments
  python3 src/force_shuffle_multi_cvae.py --split heldout --n-reassignments 10 --ckpt experiments
Writes results/force_shuffle_multi_cvae_<split>_test.json (new file).
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
sys.path.insert(0, str(HERE))
import evaluate as E
from dataset import ObjectFolderRealDataset, compute_mel_stats, load_split_indices


def ladder_cells(roots, level, split, seeds=(42, 43, 44, 45, 46)):
    """The run directories of one ladder cell, and nothing else:
    <root>/ablation_<level>_<split>_cvae_s<seed>/best_model.pt. Runs are matched
    by exact directory name."""
    out = []
    for root in roots:
        for s in seeds:
            d = Path(root) / f"ablation_{level}_{split}_cvae_s{s}"
            if (d / "best_model.pt").exists() and d not in out:
                out.append(d)
    return sorted(out, key=lambda d: d.name)


def reassignment(n, rng):
    """Random permutation with no fixed points (same construction as force_shuffle_multi.py)."""
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
def run_pass(model, perm, ctx):
    """One generation pass over the sorted test strikes. Strike at position pos
    gets the force window of test[perm[pos]]. Per strike, in this order:
    generate -> gl_inv(generated mel) -> gl_inv(real mel of this strike).
    Primed by one gl_inv call on the first test strike's real mel, which sets
    the RNG state that every generate call starts from."""
    dataset, test, device = ctx["dataset"], ctx["test"], ctx["device"]
    feat_mean, feat_std = ctx["feat_stats"]
    fwm, fws = ctx["fw_stats"]
    gl_inv, id_key = ctx["gl_inv"], ctx["id_key"]
    t0 = time.time()
    gl_inv(dataset[test[0]]["mel"])                          # priming inversion
    gens, gts = {}, {}
    for pos, i in enumerate(test):
        s = dataset[i]
        src = test[perm[pos]]
        ident = torch.tensor([s[id_key]], dtype=torch.long, device=device)
        ff = ((s["force_features"] - feat_mean) / feat_std).unsqueeze(0).to(device)
        fw = ((dataset[src]["force_waveform"] - fwm) / fws).unsqueeze(0).to(device)
        mel = model.generate(ident, ff, fw, n_samples=1)[0]
        gens[i] = gl_inv(mel)
        gts[i] = gl_inv(s["mel"])
    return dict(gen=gens, gt=gts, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["within", "heldout"], required=True)
    ap.add_argument("--n-reassignments", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=42,
                    help="reassignment seeds run from here (same draws as the DDSP run)")
    ap.add_argument("--ckpt", nargs="+", default=["experiments"],
                    help="root folders holding ablation_4_<split>_cvae_s<seed>/best_model.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[shuffle-multi-cvae] device={device} split={args.split} test split, "
          f"{args.n_reassignments} reassignments from seed {args.first_seed}")
    out = Path(args.out) if args.out else E.RESULTS_DIR / f"force_shuffle_multi_cvae_{args.split}_test.json"

    dataset = ObjectFolderRealDataset(return_waveform=False)
    split_idx = load_split_indices(dataset, str(E.SPLITS_DIR / f"splits_{args.split}.json"))
    train_idx, _, test_idx = split_idx
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
        raise SystemExit("[shuffle-multi-cvae] FAD encoder fell back; VGGish is required")

    l4_dirs = ladder_cells(args.ckpt, 4, args.split)
    print(f"[shuffle-multi-cvae] L4 checkpoints: {[d.name for d in l4_dirs]}")

    test = sorted(test_idx)
    seeds = list(range(args.first_seed, args.first_seed + args.n_reassignments))
    perms = {s: reassignment(len(test), np.random.default_rng(s)) for s in seeds}
    identity = np.arange(len(test))

    ctx = dict(dataset=dataset, test=test, device=device, feat_stats=(feat_mean, feat_std),
               fw_stats=fw_stats, gl_inv=gl_inv, id_key=None)
    seconds_per_pass = []

    # correct generations and embeddings once per checkpoint
    per_ckpt = []
    for d in l4_dirs:
        model, meta = E.load_checkpoint_and_model(d, dataset, device)
        model.eval()
        ctx["id_key"] = "obj_id" if meta["identity"] == "obj" else "material_idx"
        p_ok = run_pass(model, identity, ctx)
        seconds_per_pass.append(dict(label=f"{d.name}/correct", seconds=p_ok["seconds"]))
        print(f"[shuffle-multi-cvae] pass {d.name}/correct: {p_ok['seconds']:.1f} s", flush=True)
        objs = [dataset[i]["obj_id_raw"] for i in test]
        gt_embs = np.stack([encoder.embed(p_ok["gt"][i]) for i in test])
        ok_embs = np.stack([encoder.embed(p_ok["gen"][i]) for i in test])
        fad_ok = E.per_object_fad(ok_embs, gt_embs, objs)
        per_ckpt.append(dict(name=d.name, model=model, objs=objs, gt_embs=gt_embs, fad_ok=fad_ok))
        print(f"[shuffle-multi-cvae] {d.name}: correct FAD = {np.mean(list(fad_ok.values())):.4f}")
        del p_ok

    results = []
    for s in seeds:
        ok_acc, shuf_acc = {}, {}
        per_seed = []
        for pc in per_ckpt:
            p_sh = run_pass(pc["model"], perms[s], ctx)
            seconds_per_pass.append(dict(label=f"{pc['name']}/reassignment{s}", seconds=p_sh["seconds"]))
            print(f"[shuffle-multi-cvae] pass {pc['name']}/reassignment{s}: {p_sh['seconds']:.1f} s", flush=True)
            shuf_embs = np.stack([encoder.embed(p_sh["gen"][i]) for i in test])
            fad_shuf = E.per_object_fad(shuf_embs, pc["gt_embs"], pc["objs"])
            for o in pc["fad_ok"]:
                ok_acc.setdefault(o, []).append(pc["fad_ok"][o])
                shuf_acc.setdefault(o, []).append(fad_shuf[o])
            per_seed.append(dict(ckpt=pc["name"],
                                 fad_correct=float(np.mean(list(pc["fad_ok"].values()))),
                                 fad_shuffled=float(np.mean(list(fad_shuf.values())))))
            del p_sh
        common = sorted(set(ok_acc) & set(shuf_acc))
        delta = {o: float(np.mean(shuf_acc[o]) - np.mean(ok_acc[o])) for o in common}
        mean, ci, boot = E.paired_bootstrap_objects(delta, seed=7)
        vals = np.array([delta[o] for o in common])
        r = dict(reassignment_seed=s, n_objects=len(common), per_seed=per_seed,
                 delta_shuf_minus_ok=mean, ci95=list(ci), cohens_d_paired=E.cohens_d(vals),
                 p_two_sided=E.bootstrap_p_two_sided(boot),
                 frac_objects_worse=float(np.mean(vals > 0)))
        results.append(r)
        print(f"[shuffle-multi-cvae] reassignment {s}: delta={mean:+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"d={r['cohens_d_paired']:+.3f} p={r['p_two_sided']:.4f}")

    deltas = np.array([r["delta_shuf_minus_ok"] for r in results])
    summary = dict(n_reassignments=len(results), delta_mean=float(deltas.mean()),
                   delta_sd=float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                   delta_min=float(deltas.min()), delta_max=float(deltas.max()),
                   n_detected=int(sum(1 for r in results
                                      if r["p_two_sided"] < 0.05 and r["ci95"][0] > 0
                                      and abs(r["cohens_d_paired"]) >= 0.2)),
                   n_detected_negative=int(sum(1 for r in results
                                               if r["p_two_sided"] < 0.05 and r["ci95"][1] < 0
                                               and abs(r["cohens_d_paired"]) >= 0.2)))
    report = dict(experiment="frame_shuffle", backbone="cvae",
                  split=args.split,
                  eval_split="test", metric="object_first_VGGish_FAD", device=str(device),
                  n_test_clips=len(test), reassignment_seeds=seeds,
                  table2_reassignment_seed=42, per_reassignment=results,
                  across_reassignments=summary,
                  seconds_per_pass=seconds_per_pass, wall_seconds=time.time() - t_start,
                  test_indices=[int(i) for i in test],
                  perms={str(s): perms[s].tolist() for s in seeds})
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(E._jsonify(report), open(out, "w"), indent=2)
    print(f"[shuffle-multi-cvae] across {len(results)} reassignments: delta mean {summary['delta_mean']:+.4f} "
          f"sd {summary['delta_sd']:.4f} range [{summary['delta_min']:+.4f}, {summary['delta_max']:+.4f}] "
          f"detected in {summary['n_detected']}/{len(results)}")
    print(f"[shuffle-multi-cvae] n_detected_negative {summary['n_detected_negative']}/{len(results)}")
    print(f"[shuffle-multi-cvae] wrote {out}  (wall {time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
