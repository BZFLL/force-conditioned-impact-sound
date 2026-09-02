#!/usr/bin/env python3
"""White-noise FAD reference on the test split (descriptive).

For each split, per test object: generate as many white-noise clips
(Gaussian, 200 ms, fixed seed 0) as that object has real test clips, embed
them with the VGGish encoder and compute the same object-level FAD used for
the models. Writes results/whitenoise_fad_test.json.

Usage:
    python3 src/whitenoise_baseline.py
"""
import json
from pathlib import Path
import numpy as np
import torch, sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import evaluate as E
from dataset import ObjectFolderRealDataset, load_split_indices
from train import get_mel_stats

def main():
    device = "cpu"
    enc = E.VGGishEncoder(device)
    out = {"meta": {"status": "white-noise reference; test split",
                    "noise": "Gaussian white noise, 9600 samples, torch seed 0",
                    "encoder": "VGGishEncoder, pretrained weights (peak-norm inside embed)"}}
    ds = ObjectFolderRealDataset(return_waveform=True)
    for split in ("within", "heldout"):
        tr, va, te = load_split_indices(ds, str(ROOT / f"splits/splits_{split}.json"))
        mm, ms = get_mel_stats(ds, tr, split); ds.set_mel_stats(mm, ms)
        gt_embs, objs = [], []
        for i in te:
            s = ds[i]
            gt_embs.append(enc.embed(s["waveform"]))
            objs.append(s["obj_id_raw"])
        gt_embs = np.stack(gt_embs)
        g = torch.Generator().manual_seed(0)
        noise_embs = np.stack([enc.embed(torch.randn(9600, generator=g)) for _ in te])
        fad = E.per_object_fad(noise_embs, gt_embs, objs)
        vals = np.array(list(fad.values()))
        out[split] = {"object_first_fad_mean": float(vals.mean()),
                      "across_object_sd": float(vals.std(ddof=1)),
                      "n_objects": int(len(vals)), "n_clips": int(len(te))}
        print(f"[{split}] white-noise FAD (object-first) = {vals.mean():.3f}  (n_obj={len(vals)})")
    of = ROOT / "results/whitenoise_fad_test.json"
    json.dump(out, open(of, "w"), indent=1)
    print("wrote", of)

if __name__ == "__main__":
    main()
