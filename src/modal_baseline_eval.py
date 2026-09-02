#!/usr/bin/env python3
"""Validation-split evaluation of the SAMPLE modal-synthesis baseline.

Uses the evaluation machinery of evaluate.py (VGGish encoder,
per_object_fad, build_nn_retriever/nn_retrieve) and the modal parameters
fitted on the training split by modal_baseline_sample.py (loaded from
cache/modal_baseline_params_{split}.pt, or fitted on the spot if the cache
is missing).

What it computes (all on the validation split):
  a. within: per-object modal synthesis of every validation clip from its
     measured force window; object-level VGGish FAD against the real
     validation waveforms (embedding + per_object_fad path of
     evaluate.eval_one_model; raw waveforms, the DDSP-domain comparison
     convention).
  b. heldout: material-prior variant on the validation clips of the unseen
     objects; same scoring.
  c. nearest-force retrieval reference, both splits: the same procedure as
     evaluate.nn_lower_bound_fad but on validation indices
     (build_nn_retriever on train, retrieve for validation clips,
     backbone="ddsp" waveform domain).
  d. Ringing index (mean RMS envelope in 100-200 ms / envelope peak, mean
     over clips) for the modal synthesis on within and heldout validation
     clips, with the real recordings as anchor.
  e. Median per-clip LSD (signal_metrics.lsd), modal and retrieval.

Output: results/modal_baseline_val.json + a table on stdout.

Usage:
    python3 src/modal_baseline_eval.py
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from evaluate import (build_fad_encoder, per_object_fad, build_nn_retriever,
                      nn_retrieve, SPLITS_DIR, RESULTS_DIR)
from dataset import ObjectFolderRealDataset, load_split_indices, MATERIALS
from signal_metrics import lsd, rms_env
import modal_baseline as MB


def ring_index(w: torch.Tensor) -> float:
    """Mean RMS envelope (win=240 hop=120) in 100-200 ms / envelope peak."""
    e = rms_env(w).numpy()
    tt = np.linspace(0, 200, len(e))
    return float(e[tt >= 100].mean() / (e.max() + 1e-12))


def embed_list(encoder, wavs, tag):
    out = []
    for n, w in enumerate(wavs):
        out.append(encoder.embed(w))
        if (n + 1) % 64 == 0:
            print(f"    embed {tag}: {n + 1}/{len(wavs)}", flush=True)
    return np.stack(out)


def nmode_stats(params_by_key):
    n = np.array([len(v["freqs"]) for v in params_by_key.values()])
    return dict(n=len(n), n_modes_min=int(n.min()),
                n_modes_median=float(np.median(n)), n_modes_max=int(n.max()),
                keys_below_5_modes={str(k): int(len(v["freqs"]))
                                    for k, v in params_by_key.items()
                                    if len(v["freqs"]) < 5})


def main():
    t0 = time.time()
    print("Modal baseline (SAMPLE): evaluation on the validation split.")
    print("Parameters fitted on the training split (cached).\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                          if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device} (VGGish runs on CPU)")

    encoder, is_fallback = build_fad_encoder(device)
    if is_fallback:
        sys.exit("VGGish could not be built; a fallback-encoder FAD is not "
                 "comparable to the cited VGGish comparators.")

    ds = ObjectFolderRealDataset(return_waveform=True)

    results = {}
    for split, variant in (("within", "per-object"),
                           ("heldout", "material-prior")):
        ts = time.time()
        print(f"\n[{split}] variant = {variant}")
        cache_path = ROOT / f"cache/modal_baseline_params_{split}.pt"
        if cache_path.exists():
            params = torch.load(cache_path, map_location="cpu",
                                weights_only=False)
            print(f"  loaded fitted params from {cache_path.name} "
                  f"(fitted_on = {params['meta']['fitted_on']})")
        else:
            print("  cache missing; fitting now on the training split")
            import modal_baseline_sample
            params = modal_baseline_sample.fit_split(ds, split)

        train_idx, val_idx, test_idx = load_split_indices(
            ds, str(SPLITS_DIR / f"splits_{split}.json"))
        print(f"  n_val_clips = {len(val_idx)}")

        # modal synthesis of every validation clip from its measured force
        gens, gts, objs = [], [], []
        for n, i in enumerate(val_idx):
            s = ds[i]
            if split == "within":
                p = params["objects"][s["obj_id_raw"]]
            else:
                p = params["materials"][MATERIALS[s["material_idx"]]]
            gens.append(MB.synthesize(p, s["force_waveform"]))
            gts.append(s["waveform"])
            objs.append(str(s["obj_id_raw"]))
            if (n + 1) % 100 == 0:
                print(f"    synth: {n + 1}/{len(val_idx)}", flush=True)
        print(f"  synthesis done ({time.time() - ts:.0f}s)")

        # nearest-force retrieval reference on the validation split
        keys, tr_idx, _ = build_nn_retriever(ds, train_idx)
        retrieved_by_idx = nn_retrieve(ds, val_idx, keys, tr_idx, None, "ddsp")
        retrieved = [retrieved_by_idx[i] for i in val_idx]
        print("  retrieval done (train keys, val queries, ddsp waveform domain)")

        # embeddings + object-level FAD
        gt_embs = embed_list(encoder, gts, f"{split}/real")
        gen_embs = embed_list(encoder, gens, f"{split}/modal")
        ret_embs = embed_list(encoder, retrieved, f"{split}/retrieval")
        fad_modal = per_object_fad(gen_embs, gt_embs, objs)
        fad_retr = per_object_fad(ret_embs, gt_embs, objs)

        # ringing index + per-clip LSD
        ring = {"real": float(np.mean([ring_index(w) for w in gts])),
                "modal": float(np.mean([ring_index(w) for w in gens]))}
        lsd_modal = np.array([lsd(g, m) for g, m in zip(gts, gens)])
        lsd_retr = np.array([lsd(g, r) for g, r in zip(gts, retrieved)])

        prov = dict(fit_meta={k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in params["meta"].items()},
                    objects=nmode_stats(params["objects"]),
                    material_priors={m: int(len(v["freqs"]))
                                     for m, v in params["materials"].items()})

        results[split] = dict(
            variant=variant,
            n_val_clips=len(val_idx),
            modal=dict(object_first_fad_mean=float(np.mean(list(fad_modal.values()))),
                       n_objects=len(fad_modal),
                       fad_by_obj={k: float(v) for k, v in fad_modal.items()}),
            retrieval_val=dict(
                object_first_fad_mean=float(np.mean(list(fad_retr.values()))),
                n_objects=len(fad_retr),
                fad_by_obj={k: float(v) for k, v in fad_retr.items()}),
            ringing_index=ring,
            lsd_median_db=dict(modal=float(np.median(lsd_modal)),
                               retrieval=float(np.median(lsd_retr))),
            fitting=prov)

        r = results[split]
        print(f"  [{split}] modal FAD      = "
              f"{r['modal']['object_first_fad_mean']:.4f} "
              f"(n_obj={r['modal']['n_objects']})")
        print(f"  [{split}] retrieval FAD  = "
              f"{r['retrieval_val']['object_first_fad_mean']:.4f}")
        print(f"  [{split}] ringing real={ring['real']:.4f} "
              f"modal={ring['modal']:.4f}")
        print(f"  [{split}] LSD median: modal={r['lsd_median_db']['modal']:.2f} dB "
              f"retrieval={r['lsd_median_db']['retrieval']:.2f} dB")
        print(f"  [{split}] split done in {time.time() - ts:.0f}s", flush=True)

    print("\nSummary: object-level VGGish FAD on the validation split")
    rows = [
        ("within  modal (per-object fit)",
         results["within"]["modal"]["object_first_fad_mean"]),
        ("within  retrieval-on-val",
         results["within"]["retrieval_val"]["object_first_fad_mean"]),
        ("heldout modal (material prior)",
         results["heldout"]["modal"]["object_first_fad_mean"]),
        ("heldout retrieval-on-val",
         results["heldout"]["retrieval_val"]["object_first_fad_mean"]),
    ]
    for name, v in rows:
        print(f"  {name:<40} {v:>8.4f}")

    out = {
        "meta": {
            "status": ("SAMPLE-based modal-synthesis baseline; descriptive "
                       "reference system"),
            "eval_split": "val",
            "fit_split": "train only",
            "fad_encoder": encoder.name,
            "fad_aggregation": "object_first_perobject_fad",
            "fad_domain": ("raw waveforms, DDSP-domain comparison convention "
                           "(VGGish embed peak-normalizes per clip)"),
            "retrieval": ("nearest-force retrieval on the validation indices: "
                          "retriever built on the training split, cosine "
                          "similarity on raw force windows"),
            "ringing_definition": ("mean RMS envelope (win=240, hop=120) in "
                                   "100-200 ms divided by the envelope peak, "
                                   "mean over validation clips; real "
                                   "recordings computed on the same clips"),
            "lsd_definition": "signal_metrics.lsd (n_fft=1024, hop=256, "
                              "80 dB relative floor), median per clip",
            "method": ("Mode frequencies and decays from SAMPLE (lim-sample; "
                       "github.com/LIMUNIMI/SAMPLE, MIT; Tiraboschi, Avanzini "
                       "& Ntalampiras, SMC 2020) fitted per training recording "
                       "(peak-normalised analysis input; decays converted to "
                       "the envelope convention d = 2/decays_), aggregated "
                       "across each object's recordings by gap clustering (1% "
                       "gap clusters, >=30% support, top K=48, median "
                       "frequency/decay); SAMPLE amplitudes discarded (they "
                       "are fitted to each recording's own excitation); gains "
                       "re-fitted by joint ridge least squares against "
                       "(measured force * damped-sinusoid kernels) on the raw "
                       "training strikes; quadrature synthesis; material-prior "
                       "pooling for unseen objects."),
        },
        "within": results["within"],
        "heldout": results["heldout"],
    }
    out_path = RESULTS_DIR / "modal_baseline_val.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    main()
