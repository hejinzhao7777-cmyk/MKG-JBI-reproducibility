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
FIG.mkdir(parents=True, exist_ok=True)
CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
LAYERS = ["coexpr", "meth", "cnv"]
COLORS = {"coexpr": "#4575b4", "meth": "#d73027", "cnv": "#fdae61", "no_relation": "#6b6b6b"}

data = json.load(open(LOCK, encoding="utf-8"))
vals = {layer: [] for layer in LAYERS}
no_rel = []
for cancer in CANCERS:
    w = data[cancer]["omics_weights"]
    for layer in LAYERS:
        vals[layer].append(float(w.get(layer, 0.0)))
    no_rel.append(1.0 if data[cancer].get("weight_mode") == "reject_all_graphs" else 0.0)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(5.6, 2.9))
x = np.arange(len(CANCERS))
bottom = np.zeros(len(CANCERS))
for layer in LAYERS:
    ax.bar(x, vals[layer], bottom=bottom, label=layer, color=COLORS[layer], edgecolor="white", linewidth=0.6)
    bottom += np.asarray(vals[layer])
ax.bar(x, no_rel, bottom=bottom, label="no_relation", color=COLORS["no_relation"], edgecolor="white", linewidth=0.6)

ax.set_xticks(x)
ax.set_xticklabels(CANCERS)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Routing weight")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "Fig2_omics_weights.pdf")
fig.savefig(FIG / "Fig2_omics_weights.png", dpi=300)
print(FIG / "Fig2_omics_weights.pdf")

