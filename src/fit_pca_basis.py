#!/usr/bin/env python3
"""
Fit the PCA-6 basis for feature_set="pca6".

Train split only, per split mode, fit in pool_force_curve_np space (the L4
input space). Training and evaluation read the stored basis so that both
use the same projection.
Writes stats/pca6_basis_<split>.pt {mean(600), components(6,600), evr, meta}.

Usage:
    python3 src/fit_pca_basis.py
"""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset import ObjectFolderRealDataset, load_split_indices, pool_force_curve_np

K = 6

def main():
    ds = ObjectFolderRealDataset(return_waveform=False, feature_set="v1")
    for split in ("within", "heldout"):
        tr, va, te = load_split_indices(ds, str(HERE.parent / "splits" / f"splits_{split}.json"))
        X = np.stack([pool_force_curve_np(
            ds[i]["force_waveform"].numpy(), 32) for i in sorted(tr)])
        mean = X.mean(0)
        U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
        evr = (S ** 2 / (S ** 2).sum())
        out = HERE.parent / "stats" / f"pca6_basis_{split}.pt"
        torch.save({"mean": torch.tensor(mean),
                    "components": torch.tensor(Vt[:K]),
                    "explained_variance_ratio": torch.tensor(evr[:K]),
                    "cum_evr_k": float(evr[:K].sum()),
                    "n_train": len(tr), "split": split, "w": 32, "k": K,
                    "note": "train-split-only PCA over pool_force_curve_np space"},
                   out)
        print(f"[pca6] {split}: n_train={len(tr)}  cumEVR@{K}={evr[:K].sum():.4f}  -> {out}")

if __name__ == "__main__":
    main()
