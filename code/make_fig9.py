"""Fig.9: (A) submission-lock ablation; (B) external DCA (LIHC GSE14520)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

R = Path(os.environ.get("MKG_RESULTS_DIR", r"F:\Claude\MKG\GR第二项工作_整理\06_新实验结果"))
FIG = Path(os.environ.get("MKG_FIGURE_DIR", r"F:\Claude\MKG\GR第二项工作_整理\07_论文初稿\figures"))
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5,
                     "axes.labelsize": 7.5, "axes.titlesize": 8.0,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})

abl = json.load(open(R / "submission_lock_ablation.json", encoding="utf-8"))["_mean_exclKIRC"]
dca = json.load(open(R / "dca_results.json", encoding="utf-8"))

fig, (a, b) = plt.subplots(1, 2, figsize=(7.48, 3.1))

# A: ablation mean external C-index
schemes = [("single_coexpr", "Single\n(co-expr)"), ("equal", "Equal"),
           ("stability_only", "Stability\n-only"), ("joint", "Joint\n(MKG)")]
vals = [abl[k] for k, _ in schemes]
cols = ["#bdbdbd", "#9ecae1", "#74add1", "#2c7fb8"]
bars = a.bar(range(len(schemes)), vals, color=cols, edgecolor="white")
for i, v in enumerate(vals):
    a.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8.5)
a.set_xticks(range(len(schemes))); a.set_xticklabels([s for _, s in schemes], fontsize=8.5)
a.set_ylabel("Mean external C-index (excl. KIRC)")
a.set_ylim(0.56, 0.59)
a.set_title("(A) Submission-lock ablation", fontsize=10)
a.text(0.5, 0.565, "Frozen direction-score risk", ha="center", va="bottom", fontsize=8, color="#555555")

# B: DCA
thr = np.array(dca["thresholds"]); nb = dca["net_benefit"]
styles = {"MKG": ("#2c7fb8", "-", 2.2), "Cox-Lasso": ("#d95f0e", "-", 1.8),
          "Treat all": ("#888888", "--", 1.3), "Treat none": ("#000000", ":", 1.2)}
for k in ["MKG", "Cox-Lasso", "Treat all", "Treat none"]:
    col, ls, lw = styles[k]
    b.plot(thr, np.array(nb[k]), ls, color=col, lw=lw, label=k)
b.axhline(0, color="grey", lw=0.5)
b.set_xlabel("Threshold probability"); b.set_ylabel("Net benefit")
b.set_xlim(0.05, 0.6)
model_values = np.concatenate([np.asarray(nb["MKG"]), np.asarray(nb["Cox-Lasso"])])
b.set_ylim(float(model_values.min()) - 0.01, float(model_values.max()) + 0.01)
b.set_title("(B) Decision curve (LIHC external, 3-yr)", fontsize=10); b.legend(frameon=False, fontsize=8.5)

fig.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.90, wspace=0.30)
for ext in ["pdf", "png", "svg", "tiff"]:
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
    if ext in {"png", "tiff"}:
        kwargs["dpi"] = 600
    fig.savefig(FIG / f"Fig9_ablation_dca.{ext}", **kwargs)
print("saved Fig9_ablation_dca")
