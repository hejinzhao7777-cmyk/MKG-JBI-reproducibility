import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "07_论文初稿" / "figures" / "Fig1_MKG_complete_framework.pdf"
OUT_SVG = ROOT / "07_论文初稿" / "figures" / "Fig1_MKG_complete_framework.svg"


def box(ax, xy, wh, text, fc, ec="#4A5568", fs=8.5, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(patch)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        yy = y + h * (0.55 + 0.18 * (len(lines) - 1) / 2 - 0.18 * i)
        ax.text(x + w / 2, yy, line, ha="center", va="center", fontsize=fs, weight=weight)
    return patch


def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.1, color="#4A5568"))


def main():
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(15.5, 8.8))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.8)
    ax.axis("off")

    ax.text(7.75, 8.45, "MKG: Multi-omics Knowledge-Guided Graph Routing Framework",
            ha="center", va="center", fontsize=18, weight="bold")
    ax.text(7.75, 8.12,
            "Relation reliability routing, stable signature selection, frozen molecular scoring, and IPCW conformal uncertainty",
            ha="center", va="center", fontsize=10, color="#607D8B")

    panels = [
        (0.25, 0.75, 2.65, 6.95, "#E8F3FB", "(a) Inputs and graphs"),
        (3.1, 0.75, 3.05, 6.95, "#EAF7EF", "(b) Relation routing"),
        (6.35, 0.75, 3.05, 6.95, "#EAF7EF", "(c) Stable selection"),
        (9.6, 0.75, 2.65, 6.95, "#FFF1DF", "(d) Risk scoring"),
        (12.45, 0.75, 2.8, 6.95, "#F1E8FA", "(e) IPCW conformal"),
    ]
    for x, y, w, h, fc, title in panels:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
                                    linewidth=1.0, edgecolor="#CBD5E0", facecolor=fc))
        ax.text(x + 0.15, y + h - 0.35, title, ha="left", va="center", fontsize=11, weight="bold")

    box(ax, (0.55, 6.35), (0.95, 0.55), "RNA-seq\nexpression", "#D8EAF8")
    box(ax, (0.55, 5.55), (0.95, 0.55), "DNA\nmethylation", "#D8EAF8")
    box(ax, (0.55, 4.75), (0.95, 0.55), "Copy-number\nprofiles", "#D8EAF8")
    box(ax, (0.55, 3.75), (0.95, 0.55), "OS time and\ncensoring", "#FDEAEA")
    box(ax, (1.75, 5.65), (0.9, 1.05), "Preprocess:\nstandardize,\nmap to genes", "#FFFFFF", fs=7.5)
    box(ax, (1.75, 4.0), (0.9, 0.75), "Null Cox\ndeviance\nresidual D", "#FFFFFF", fs=7.5)
    box(ax, (0.65, 1.75), (1.95, 1.4), "Three normalized\nmolecular graphs:\nco-expression,\nmethylation-expression,\nCNV co-perturbation", "#DDF1E4", fs=8)
    arrow(ax, (1.5, 6.6), (1.75, 6.25))
    arrow(ax, (1.5, 5.8), (1.75, 6.05))
    arrow(ax, (1.5, 5.0), (1.75, 5.85))
    arrow(ax, (1.5, 4.0), (1.75, 4.35))
    arrow(ax, (2.2, 5.65), (1.6, 3.15))
    arrow(ax, (2.2, 4.0), (1.8, 3.15))

    box(ax, (3.35, 6.25), (2.55, 0.8), "For each graph layer k:\n30 bootstraps and Top-K lists", "#FFFFFF")
    box(ax, (3.35, 5.05), (1.2, 0.8), "Selection\nstability S_k\nmean RBO@20", "#DDF1E4", fs=7.5)
    box(ax, (4.7, 5.05), (1.2, 0.8), "Prediction gain\nDelta_k vs\nL0 = 0", "#DDF1E4", fs=7.5)
    box(ax, (3.55, 3.75), (2.15, 0.9), "Joint utility routing:\nw_k proportional to\nS_k x max(Delta_k, 0)", "#CDEBD5")
    box(ax, (3.55, 2.35), (2.15, 0.85), "No-relation option:\nreject all graphs if\nall gains are non-positive", "#FFFFFF", fs=8)
    ax.text(4.62, 1.3, "A graph contributes only if it is\nreproducible and improves ranking.",
            ha="center", va="center", fontsize=8, color="#607D8B")
    arrow(ax, (4.62, 6.25), (3.95, 5.85))
    arrow(ax, (4.62, 6.25), (5.3, 5.85))
    arrow(ax, (3.95, 5.05), (4.2, 4.65))
    arrow(ax, (5.3, 5.05), (5.05, 4.65))
    arrow(ax, (4.62, 3.75), (4.62, 3.2))

    box(ax, (6.65, 6.1), (2.4, 0.9), "Graph-Lasso on deviance\nresiduals: sparse coefficients\nplus graph smoothness", "#FFFFFF", fs=8)
    box(ax, (6.65, 4.8), (1.05, 0.8), "PGD solver\nwith soft\nthresholding", "#DDF1E4", fs=7.2)
    box(ax, (8.0, 4.8), (1.05, 0.8), "Random forest\nnonlinear\nimportance", "#DDF1E4", fs=7.2)
    box(ax, (6.85, 3.45), (2.0, 0.85), "QP engine fusion +\nempirical-CDF rank fusion", "#FFFFFF", fs=8)
    box(ax, (7.05, 2.05), (1.6, 0.9), "Top-20 stable\ngene signature\nand frequencies", "#CDEBD5", fs=8)
    arrow(ax, (4.62, 2.35), (6.35, 2.5))
    arrow(ax, (7.85, 6.1), (7.2, 5.6))
    arrow(ax, (7.85, 6.1), (8.55, 5.6))
    arrow(ax, (7.2, 4.8), (7.45, 4.3))
    arrow(ax, (8.55, 4.8), (8.1, 4.3))
    arrow(ax, (7.85, 3.45), (7.85, 2.95))

    box(ax, (9.9, 6.1), (2.0, 0.8), "Frozen molecular\nrisk score from\nTop-20 genes", "#FFE3C2")
    box(ax, (9.9, 4.8), (2.0, 0.8), "MKG risk score r(x):\ndirections x weights;\nexternal frozen", "#FFFFFF", fs=7.8)
    box(ax, (9.9, 3.45), (2.0, 0.8), "Random survival forest\nestimates S(t|x)\nfor conformal scores", "#FFFFFF", fs=7.8)
    box(ax, (10.0, 2.05), (1.8, 0.9), "Selection optimized\nfor stability;\nrisk groups from r(x)", "#FFF8EF", fs=7.8)
    arrow(ax, (9.4, 2.5), (9.9, 2.5))
    arrow(ax, (10.9, 6.1), (10.9, 5.6))
    arrow(ax, (10.9, 4.8), (10.9, 4.25))
    arrow(ax, (12.25, 3.85), (12.45, 3.85))
    arrow(ax, (11.9, 5.2), (12.75, 4.15))

    box(ax, (12.75, 6.25), (2.15, 0.75), "Train / calibration /\ntest split; alpha = 0.10", "#FFFFFF")
    box(ax, (12.75, 5.0), (2.15, 0.9), "IPCW-weighted\nhorizon score R_i(t)\nfor known labels", "#E5D7F3", fs=7.8)
    box(ax, (12.75, 3.75), (2.15, 0.8), "Risk-group IPCW\nquantile gives pointwise\npredictive interval", "#E5D7F3", fs=7.8)
    box(ax, (12.75, 2.2), (2.15, 0.95), "Outputs: signature,\nrisk score, survival curve,\npointwise interval", "#D9C8EE", fs=7.8)
    arrow(ax, (13.82, 6.25), (13.82, 5.9))
    arrow(ax, (13.82, 5.0), (13.82, 4.55))
    arrow(ax, (13.82, 3.75), (13.82, 3.15))

    ax.add_patch(FancyBboxPatch((0.4, 0.15), 14.7, 0.45, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.0, edgecolor="#CBD5E0", facecolor="#F7F8FA"))
    ax.text(7.75, 0.38,
            "Theory map: Prop. 1 surrogate support recovery | Prop. 2 joint utility routing properties | Prop. 3 horizon-specific IPCW conformal calibration",
            ha="center", va="center", fontsize=9)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")


if __name__ == "__main__":
    main()

