import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOCK_DIR = Path(os.environ.get("MKG_LOCK_DIR", ROOT / "09_JBI_SUBMISSION_LOCK"))
LOCK = LOCK_DIR / "final_config_comparison_ALL_LOCKED.json"
FIG = ROOT / "07_论文初稿" / "figures"
CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
METHODS = ["GR-SAFS_v2", "Uni-Cox", "Cox-Lasso", "Cox-EN", "RSF", "DeepSurv"]
LABELS = {"GR-SAFS_v2": "MKG"}
COLORS = {
    "GR-SAFS_v2": "#2c7fb8",
    "Uni-Cox": "#756bb1",
    "Cox-Lasso": "#d95f0e",
    "Cox-EN": "#e6550d",
    "RSF": "#777777",
    "DeepSurv": "#999999",
}

data = json.load(open(LOCK, encoding="utf-8"))

def ext_metric(cancer, method):
    ds = next(iter(data[cancer]["external"]))
    return data[cancer]["external"][ds][method]["c_index"]

def train_metric(cancer, method):
    return data[cancer]["training"][method]["c_index"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), gridspec_kw={"width_ratios": [1.05, 1.25]})

non_kirc = [c for c in CANCERS if c != "KIRC"]
y = np.arange(len(METHODS))
for i, method in enumerate(METHODS):
    tr = np.mean([train_metric(c, method) for c in non_kirc])
    ex = np.mean([ext_metric(c, method) for c in non_kirc])
    color = COLORS[method]
    axes[0].plot([tr, ex], [i, i], color=color, lw=1.6)
    axes[0].scatter([tr], [i], color="white", edgecolor=color, s=42, zorder=3)
    axes[0].scatter([ex], [i], color=color, s=42, marker="D", zorder=3)
    axes[0].text(min(tr, ex) - 0.006, i, f"{tr-ex:+.2f}", ha="right", va="center", fontsize=7, color=color)

axes[0].set_yticks(y)
axes[0].set_yticklabels([LABELS.get(m, m) for m in METHODS])
axes[0].invert_yaxis()
axes[0].axvline(0.5, color="#888888", ls=":", lw=0.8)
axes[0].set_xlabel("C-index")
axes[0].set_title("A. Internal OOF to external transfer\n(mean, excluding KIRC)")
axes[0].set_xlim(0.38, 0.86)

x = np.arange(len(CANCERS))
width = 0.24
for j, method in enumerate(["GR-SAFS_v2", "Uni-Cox", "Cox-Lasso"]):
    axes[1].bar(
        x + (j - 1) * width,
        [ext_metric(c, method) for c in CANCERS],
        width,
        label=LABELS.get(method, method),
        color=COLORS[method],
    )
axes[1].axhline(0.5, color="#888888", ls=":", lw=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(CANCERS)
axes[1].set_ylabel("External C-index")
axes[1].set_ylim(0.2, 0.72)
axes[1].set_title("B. Per-cancer external discrimination")
axes[1].legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))

fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "Fig3_generalization.pdf")
fig.savefig(FIG / "Fig3_generalization.png", dpi=300)
print(FIG / "Fig3_generalization.pdf")

