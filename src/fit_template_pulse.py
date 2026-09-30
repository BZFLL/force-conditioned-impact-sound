#!/usr/bin/env python3
"""
Fit the template pulse of the template-pulse control.

For one split, every training strike's measured 200 ms force window (the raw
L4 input) is divided by its own max|window|, and the normalised windows are
averaged. The template-pulse control feeds the DDSP L4 per-frame route
template * max|window of strike i| instead of the measured window
(dataset.py, force_input="template_peak").

Train split only. The file is part of the repository and is not refit, as for
the PCA bases: training and evaluation use the same template on every machine.
Writes stats/template_pulse_<split>.pt {mean_pulse (9600,) float64, n_train,
split, peak_index, peak_index_expected, n_peak_mismatch} and
stats/template_pulse_<split>.png (first 30 ms and the full 200 ms).

The fit asserts argmax|mean_pulse| lies within 2 samples of 240 (the windows
start 5 ms before the force peak). n_peak_mismatch counts training strikes whose
window max|.| differs from the descriptor's whole-recording peak
(force_features[0]) by more than 1e-6 relative; the count is reported, not
fixed.

Usage:
    python3 src/fit_template_pulse.py --split within
    python3 src/fit_template_pulse.py --split heldout
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from dataset import ObjectFolderRealDataset, load_split_indices

PEAK_INDEX_EXPECTED = 240   # 5 ms at 48 kHz
PEAK_INDEX_TOL = 2
PEAK_REL_TOL = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["within", "heldout"], required=True)
    split = ap.parse_args().split
    out = ROOT / "stats" / f"template_pulse_{split}.pt"
    png = ROOT / "stats" / f"template_pulse_{split}.png"
    if out.exists():
        raise SystemExit(f"[template] {out} exists; the template is not refit")

    ds = ObjectFolderRealDataset()
    train_idx, _, _ = load_split_indices(ds, str(ROOT / "splits" / f"splits_{split}.json"))

    normed = []
    n_mismatch = 0
    for i in train_idx:
        w = ds[i]["force_waveform"].numpy().astype(np.float64)
        peak = float(np.abs(w).max())
        assert peak > 0, f"strike {i}: all-zero force window"
        normed.append(w / peak)
        desc_peak = float(ds[i]["force_features"][0])
        if abs(peak - desc_peak) > PEAK_REL_TOL * abs(desc_peak):
            n_mismatch += 1
    mean_pulse = np.stack(normed).mean(axis=0)
    peak_index = int(np.abs(mean_pulse).argmax())

    print(f"[template] {split}: n_train={len(train_idx)} peak_index={peak_index} "
          f"(expected {PEAK_INDEX_EXPECTED} +/- {PEAK_INDEX_TOL}) "
          f"value at peak={mean_pulse[peak_index]!r} n_peak_mismatch={n_mismatch}")
    assert abs(peak_index - PEAK_INDEX_EXPECTED) <= PEAK_INDEX_TOL, \
        f"template peak at sample {peak_index}, expected {PEAK_INDEX_EXPECTED}"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t_ms = np.arange(len(mean_pulse)) / 48.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    n30 = int(30 * 48)
    axes[0].plot(t_ms[:n30], mean_pulse[:n30], lw=1)
    axes[0].set_title(f"template pulse, {split}: first 30 ms")
    axes[1].plot(t_ms, mean_pulse, lw=0.8)
    axes[1].set_title(f"full 200 ms (n_train={len(train_idx)})")
    for ax in axes:
        ax.set_xlabel("time (ms)")
        ax.axvline(PEAK_INDEX_EXPECTED / 48.0, color="grey", lw=0.5, ls="--")
    axes[0].set_ylabel("mean of window / max|window|")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)

    torch.save({"mean_pulse": torch.tensor(mean_pulse, dtype=torch.float64),
                "n_train": len(train_idx), "split": split,
                "peak_index": peak_index, "peak_index_expected": PEAK_INDEX_EXPECTED,
                "n_peak_mismatch": n_mismatch}, out)
    print(f"[template] wrote {out}")
    print(f"[template] wrote {png}")


if __name__ == "__main__":
    main()
