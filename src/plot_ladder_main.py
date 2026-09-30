#!/usr/bin/env python3
"""Fig. 2 (rows L2-L4): change in object-level FAD relative to L1 on the test split.

Paired-difference dot plot. Left/right panels = seen objects (within split)
/ unseen objects (held-out split); x-axis = L2/L3/L4; y-axis = FAD(level)
minus FAD(L1) (negative = better); error bars = 95% bootstrap CI of the
paired per-object difference; dashed line at 0 = the L1 baseline. A CI that
excludes 0 corresponds to the significance markers in Table 1.

Inputs: results/ladder_ddsp_{within,heldout}_test.json (vs_L1) and
results/results_cvae_{within,heldout}_test.json (levels + primary_comparisons).
Output: figures/fig_ladder_main.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

COLOR = {"ddsp": "#2a78d6", "cvae": "#eb6834"}  # categorical colour pair
MARKER = {"ddsp": "o", "cvae": "s"}
LABEL = {"ddsp": "DDSP", "cvae": "CVAE"}
INK = "#0b0b0b"
INK2 = "#52514e"


def ddsp_cell(split):
    d = json.load(open(ROOT / f"results/ladder_ddsp_{split}_test.json"))
    head = d["headline_object_first_fad"]
    vs = d["vs_L1"]
    levels = []
    for want in ("L1", "L2a", "L3a", "L4"):
        key = want if want in head else want.rstrip("a")
        levels.append(head[key])
    deltas, cis = [], []
    for want in ("L2a-L1", "L3a-L1", "L4-L1"):
        key = want if want in vs else want.replace("a-", "-")
        deltas.append(vs[key]["delta"])
        cis.append(vs[key]["ci95"])
    return levels, deltas, cis


def cvae_cell(split):
    d = json.load(open(ROOT / f"results/results_cvae_{split}_test.json"))
    lv = d["levels"]
    levels = [lv[str(i)]["object_first_fad_mean"] for i in (1, 2, 3, 4)]
    pc = d["primary_comparisons"]
    deltas = [levels[i] - levels[0] for i in (1, 2, 3)]
    cis = [pc[str(i)]["ci95"] for i in (2, 3, 4)]
    return levels, deltas, cis


data = {}
for split in ("within", "heldout"):
    for arch, fn in (("ddsp", ddsp_cell), ("cvae", cvae_cell)):
        levels, deltas, cis = fn(split)
        data[(arch, split)] = (deltas, cis)

plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.edgecolor": INK2,
})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True,
                         constrained_layout=True)
PANEL = {"within": "Seen objects", "heldout": "Unseen objects"}
XT = ["L2\n(peak)", "L3\n(descriptor)", "L4\n(frames)"]

for ax, split in zip(axes, ("within", "heldout")):
    ax.axhline(0.0, color=INK2, lw=0.8, ls="--", zorder=1)
    for arch, off in (("ddsp", -0.09), ("cvae", 0.09)):
        deltas, cis = data[(arch, split)]
        xs = [i + off for i in range(3)]
        lo = [d - c[0] for d, c in zip(deltas, cis)]
        hi = [c[1] - d for d, c in zip(deltas, cis)]
        ax.errorbar(xs, deltas, yerr=[lo, hi], fmt=MARKER[arch], ms=4.5,
                    color=COLOR[arch], ecolor=COLOR[arch], elinewidth=1.2,
                    capsize=2.5, capthick=1.2, lw=0, zorder=3,
                    label=LABEL[arch])
    ax.set_title(PANEL[split], fontsize=8.5, color=INK)
    ax.set_xticks(range(3), XT)
    ax.set_xlim(-0.5, 2.5)
    ax.grid(axis="y", color=INK2, alpha=0.18, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

axes[0].set_ylabel("FAD change vs. L1 (identity)\n(negative = better)")
axes[0].annotate("L1", xy=(2.44, 0), xytext=(2.44, 0.035),
                 fontsize={True: 7, False: 7}[True], color=INK2, ha="right")
axes[0].legend(loc="upper right", frameon=False, fontsize=8,
               handletextpad=0.4, borderaxespad=0.2)
# direct labels at the end of each series in the right panel
dd, cc = data[("ddsp", "heldout")], data[("cvae", "heldout")]
axes[1].annotate("DDSP", xy=(2 - 0.09, dd[0][2]), xytext=(1.62, dd[0][2] - 0.02),
                 fontsize=7, color=INK, ha="right", va="center")
axes[1].annotate("CVAE", xy=(2 + 0.09, cc[0][2]), xytext=(2.38, cc[0][2] + 0.10),
                 fontsize=7, color=INK, ha="left", va="center")

out = OUT / "fig_ladder_main.png"
fig.savefig(out, dpi=300)
print(f"written: {out}")
