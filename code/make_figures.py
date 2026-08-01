"""
生成 MKG 论文主图(英文、无烤入'Fig.N'标题,题注交给 LaTeX)。
Fig2/3/4/7 由结果 JSON 重绘;Fig5(患者)/Fig6(敏感性) 从 JSON 英文重绘(不重算)。
Fig1 框架图正文用 LaTeX 占位,这里不再需要。
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
R = Path(os.environ.get(
    "MKG_RESULTS_DIR",
    PROJECT / "outputs",
))
FIG = Path(os.environ.get(
    "MKG_FIGURE_DIR",
    PROJECT / "results" / "generated_figures",
))
FIG.mkdir(parents=True, exist_ok=True)
CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
DS1 = {"LUAD": "GSE31210", "LIHC": "GSE14520", "KIRC": "GSE29609",
       "COAD": "GSE39582", "STAD": "GSE84437", "HNSC": "GSE65858"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
C_MKG, C_BASE = "#2c7fb8", "#d95f0e"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig); print("saved", name)


def fig2():
    comp = {c: json.load(open(R / f"final_config_comparison_{c}.json", encoding="utf-8"))[c] for c in CANCERS}
    w = {c: comp[c]["omics_weights"] for c in CANCERS}
    layers, cols = ["coexpr", "meth", "cnv"], ["#4575b4", "#d73027", "#fdae61"]
    fig, ax = plt.subplots(figsize=(7, 3.6)); bottom = np.zeros(len(CANCERS))
    for L, col in zip(layers, cols):
        vals = [w[c][L] for c in CANCERS]
        ax.bar(CANCERS, vals, bottom=bottom, label=L, color=col, edgecolor="white"); bottom += vals
    ax.set_ylabel("Omics graph weight"); ax.set_ylim(0, 1.0)
    ax.legend(title="Layer", frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "Fig2_omics_weights")


def fig3():
    rp = json.load(open(R / "refit_predictor_results.json", encoding="utf-8"))
    methods = ["GR-SAFS_v2", "Uni-Cox", "Cox-Lasso", "Cox-EN", "CGBoost", "RSF"]
    lab = {"GR-SAFS_v2": "MKG", "Uni-Cox": "Uni-Cox", "Cox-Lasso": "Cox-Lasso",
           "Cox-EN": "Cox-EN", "CGBoost": "CGBoost", "RSF": "RSF"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0))
    cex = [c for c in CANCERS if c != "KIRC"]
    for i, m in enumerate(methods):
        oof = np.mean([rp[c][m]["train_oof_refit"] for c in cex])
        ext = np.mean([rp[c][m]["external"][DS1[c]]["refit_c_index"] for c in cex])
        col = C_MKG if m == "GR-SAFS_v2" else "#999999"
        a.plot([oof, ext], [i, i], "-", color=col, lw=2, zorder=1)
        a.scatter([oof], [i], color="#888", s=55, zorder=2, label="OOF (internal)" if i == 0 else "")
        a.scatter([ext], [i], color=col, s=70, marker="D", zorder=3, label="External" if i == 0 else "")
        a.text(min(oof, ext) - 0.006, i, f"$\\Delta$={oof-ext:+.2f}", ha="right", va="center", fontsize=8, color=col)
    a.set_yticks(range(len(methods))); a.set_yticklabels([lab[m] for m in methods])
    a.set_xlabel("C-index"); a.axvline(0.5, color="grey", lw=0.6, ls=":")
    a.set_title("(A) OOF vs external (mean, excl. KIRC) & decay", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="lower right")
    keym, cols = ["GR-SAFS_v2", "Uni-Cox", "Cox-Lasso"], [C_MKG, "#756bb1", "#d95f0e"]
    x = np.arange(len(CANCERS)); wb = 0.26
    for j, (m, col) in enumerate(zip(keym, cols)):
        b.bar(x + (j-1)*wb, [rp[c][m]["external"][DS1[c]]["refit_c_index"] for c in CANCERS], wb, label=lab[m], color=col)
    b.axhline(0.5, color="grey", lw=0.6, ls=":"); b.set_xticks(x); b.set_xticklabels(CANCERS)
    b.set_ylabel("External C-index"); b.set_ylim(0, 0.75); b.set_title("(B) Per-cancer external C-index", fontsize=10)
    b.legend(frameon=False, fontsize=8); b.text(2, 0.30, "KIRC: under-powered\n(n=39, Agilent)", fontsize=7, ha="center", color="#d73027")
    save(fig, "Fig3_generalization")


def fig4():
    d = json.load(open(R / "stability_and_weighted_results.json", encoding="utf-8"))
    mkg = [d[c]["stability"]["GR-SAFS"]["RBO"] for c in CANCERS]
    las = [d[c]["stability"]["Cox-Lasso"]["RBO"] for c in CANCERS]
    mkgj = [d[c]["stability"]["GR-SAFS"]["Jaccard"] for c in CANCERS]
    lasj = [d[c]["stability"]["Cox-Lasso"]["Jaccard"] for c in CANCERS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0)); x = np.arange(len(CANCERS)); w = 0.38
    for ax, mk, ls, ttl in [(a, mkg, las, "(A) Rank-biased overlap (RBO)"), (b, mkgj, lasj, "(B) Jaccard overlap")]:
        ax.bar(x - w/2, mk, w, label="MKG", color=C_MKG); ax.bar(x + w/2, ls, w, label="Cox-Lasso", color=C_BASE)
        for i in range(len(CANCERS)):
            ax.text(x[i], max(mk[i], ls[i]) + 0.01, f"{mk[i]/ls[i]:.1f}x", ha="center", fontsize=8, color=C_MKG)
        ax.set_xticks(x); ax.set_xticklabels(CANCERS); ax.set_ylabel("Bootstrap reproducibility")
        ax.set_ylim(0, 0.42); ax.set_title(ttl, fontsize=10); ax.legend(frameon=False)
    save(fig, "Fig4_stability")


def fig5():
    """患者 Conformal 带(英文,从 patient_cases.json)"""
    d = json.load(open(R / "patient_cases.json", encoding="utf-8"))
    cases = d["cases"]; ymap = {"1年": 1, "3年": 3, "5年": 5}
    grp = {"低风险": "Low risk", "中风险": "Mid risk", "高风险": "High risk"}
    fig, axes = plt.subplots(1, len(cases), figsize=(2.6*len(cases), 3.0), sharey=True)
    if len(cases) == 1: axes = [axes]
    for ax, c in zip(axes, cases):
        yrs = [1, 3, 5]
        pts = [c["conformal_band"][k]["point"] for k in ["1年", "3年", "5年"]]
        lo = [c["conformal_band"][k]["point"] - c["conformal_band"][k]["lower"] for k in ["1年", "3年", "5年"]]
        hi = [c["conformal_band"][k]["upper"] - c["conformal_band"][k]["point"] for k in ["1年", "3年", "5年"]]
        ax.errorbar(yrs, pts, yerr=[lo, hi], fmt="o-", color=C_MKG, capsize=4, ms=5, lw=1.5)
        et = c["observed_OS_time_days"]/365.0
        ax.axvline(et, ls="--", color="grey", lw=1)
        ax.set_title(f"{grp.get(c['risk_group'], c['risk_group'])}\n(P{c['risk_percentile']})", fontsize=9)
        ax.set_xlabel("Years"); ax.set_xticks([1, 3, 5]); ax.set_ylim(0, 1.02); ax.set_xlim(0, 6)
    axes[0].set_ylabel("Survival probability")
    save(fig, "Fig5_conformal")


def fig6():
    """超参敏感性(英文,从 sensitivity_results.json)"""
    d = json.load(open(R / "sensitivity_results.json", encoding="utf-8"))
    old = plt.rcParams.copy()
    plt.rcParams.update({"font.family": "Arial", "font.size": 7.5,
                         "axes.labelsize": 7.5, "axes.titlesize": 8.0,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(2, 2, figsize=(7.48, 5.15), sharey=True)
    panels = [
        ("lambda1", r"Sparsity penalty $\lambda_1$", r"$\lambda_1$", ax[0, 0]),
        ("lambda2", r"Graph penalty $\lambda_2$", r"$\lambda_2$", ax[0, 1]),
        ("top_k", "Signature size K", "K", ax[1, 0]),
        ("bootstrap_B", "Subsampling repetitions B", "B", ax[1, 1]),
    ]
    for panel_label, (key, title, xlabel, a) in zip(("(A)", "(B)", "(C)", "(D)"), panels):
        xs = [e["val"] for e in d[key]]
        a.plot(xs, [e["train"] for e in d[key]], "o-", color=C_MKG, label="Train (OOF)")
        a.plot(xs, [e["ext"] for e in d[key]], "s--", color=C_BASE, label="External (GSE31210)")
        a.axhline(0.5, color="grey", lw=0.6, ls=":")
        a.set_xlabel(xlabel); a.set_ylabel("C-index"); a.set_title(title, loc="left")
        a.set_ylim(0.49, 0.79)
        a.text(-0.13, 1.05, panel_label, transform=a.transAxes,
               fontsize=8.5, fontweight="bold")
        if key == "lambda2":
            a.set_xscale("log")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=7.0, bbox_to_anchor=(0.55, 0.995))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.10, top=0.90,
                        wspace=0.24, hspace=0.37)
    fig.savefig(FIG / "Fig6_sensitivity.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / "Fig6_sensitivity.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / "Fig6_sensitivity.svg", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    plt.rcParams.update(old)
    print("saved Fig6_sensitivity")


def fig7():
    data = [("LUAD", "Wnt signaling", 0.012), ("KIRC", "JAK-STAT", 0.012),
            ("KIRC", "Ferroptosis", 0.040), ("COAD", "B-cell receptor", 0.003), ("COAD", "PI3K-AKT", 0.048)]
    cancers = ["LUAD", "KIRC", "COAD"]; paths = list(dict.fromkeys([p for _, p, _ in data]))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for c, p, pv in data:
        ax.scatter(cancers.index(c), paths.index(p), s=-np.log10(pv)*130,
                   c=[-np.log10(pv)], cmap="YlOrRd", vmin=1, vmax=3, edgecolor="k", zorder=3)
    ax.set_xticks(range(len(cancers))); ax.set_xticklabels(cancers)
    ax.set_yticks(range(len(paths))); ax.set_yticklabels(paths)
    ax.set_xlim(-0.5, len(cancers)-0.5); ax.set_ylim(-0.5, len(paths)-0.5)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(1, 3))
    fig.colorbar(sm, ax=ax, label="-log10(p)")
    save(fig, "Fig7_kegg")


if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("DONE ->", FIG)
