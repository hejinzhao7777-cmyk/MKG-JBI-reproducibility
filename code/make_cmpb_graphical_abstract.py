#!/usr/bin/env python
"""Generate the submission graphical abstract as an editable vector figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
MAGENTA = "#CC79A7"
DARK = "#1F2933"
GREY = "#6B7280"
LIGHT_BLUE = "#EAF4FB"
LIGHT_GREEN = "#EAF7F1"
LIGHT_ORANGE = "#FFF4DF"
LIGHT_PURPLE = "#F7EEF7"


def box(ax, xy, width, height, facecolor, edgecolor=DARK, radius=0.12, lw=1.2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.03,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=DARK, lw=1.5, style="-|>", mutation=13):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def draw_network(ax, cx, cy, color, dashed=False):
    points = [
        (cx - 0.23, cy + 0.04),
        (cx - 0.08, cy + 0.18),
        (cx + 0.10, cy + 0.12),
        (cx + 0.23, cy - 0.05),
        (cx - 0.05, cy - 0.16),
        (cx + 0.08, cy - 0.02),
    ]
    edges = [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 5), (3, 4), (4, 5)]
    for i, j in edges:
        ax.plot(
            [points[i][0], points[j][0]],
            [points[i][1], points[j][1]],
            color=color,
            linewidth=1.1,
            linestyle="--" if dashed else "-",
            alpha=0.9,
        )
    for x, y in points:
        ax.add_patch(Circle((x, y), 0.045, facecolor=color, edgecolor="white", linewidth=0.6))


def draw_gene_list(ax, x, y):
    lengths = [0.42, 0.34, 0.48, 0.28, 0.39]
    colors = [BLUE, GREEN, ORANGE, MAGENTA, VERMILLION]
    for i, (length, color) in enumerate(zip(lengths, colors)):
        yy = y - i * 0.12
        ax.add_patch(Circle((x, yy), 0.035, facecolor=color, edgecolor="white", linewidth=0.5))
        ax.plot([x + 0.08, x + 0.08 + length], [yy, yy], color=GREY, linewidth=2.0)


def make_figure(output_base: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    card_x = [0.18, 2.62, 5.20, 7.70]
    card_w = [2.05, 2.18, 2.10, 2.12]
    faces = [LIGHT_BLUE, LIGHT_GREEN, LIGHT_ORANGE, LIGHT_PURPLE]
    headers = [
        "1  Candidate relation layers",
        "2  Reliability gate",
        "3  Compact frozen signature",
        "4  Outcome-free external test",
    ]
    for x, w, face, header in zip(card_x, card_w, faces, headers):
        box(ax, (x, 0.35), w, 3.30, face, radius=0.18, lw=1.4)
        ax.text(
            x + w / 2,
            3.38,
            header,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=DARK,
        )

    # Card 1: heterogeneous relation layers and the explicit zero graph.
    layer_y = [2.72, 2.05, 1.38]
    labels = ["RNA co-expression", "Methylation-\nexpression", "Copy-number\nco-perturbation"]
    colors = [BLUE, MAGENTA, ORANGE]
    for cy, label, color in zip(layer_y, labels, colors):
        draw_network(ax, 0.72, cy, color)
        ax.text(1.06, cy, label, va="center", ha="left", fontsize=7.8, color=DARK)
    ax.plot([0.42, 0.95], [0.73, 0.73], color=GREY, linewidth=1.3, linestyle="--")
    ax.text(1.06, 0.73, "Zero graph $L_0=0$", va="center", ha="left", fontsize=8.5, color=DARK)

    # Card 2: paired stability and utility evidence, followed by honest branching.
    box(ax, (2.86, 2.48), 0.77, 0.52, "white", edgecolor=BLUE, radius=0.08)
    box(ax, (3.78, 2.48), 0.77, 0.52, "white", edgecolor=GREEN, radius=0.08)
    ax.text(3.245, 2.74, "Bootstrap\nstability $S_k$", ha="center", va="center", fontsize=8.6)
    ax.text(4.165, 2.74, "OOF gain\n$\\Delta_k$ vs $L_0$", ha="center", va="center", fontsize=8.6)
    arrow(ax, (3.64, 2.74), (3.76, 2.74), color=GREY, lw=1.1, mutation=10)
    ax.text(
        3.70,
        2.14,
        "$u_k=S_k\\,\\max(\\Delta_k,0)$",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color=DARK,
    )
    arrow(ax, (3.70, 2.44), (3.70, 2.30), color=DARK, lw=1.2)
    box(ax, (2.88, 1.08), 0.78, 0.62, "white", edgecolor=GREEN, radius=0.08)
    box(ax, (3.75, 1.08), 0.78, 0.62, "white", edgecolor=GREY, radius=0.08)
    ax.text(
        3.27,
        1.39,
        "$\\sum u_k>0$\nEligible graphs\nweighted",
        ha="center",
        va="center",
        fontsize=7.5,
    )
    ax.text(
        4.14,
        1.39,
        "$\\sum u_k=0$\nAll graphs\nrejected",
        ha="center",
        va="center",
        fontsize=7.5,
    )
    arrow(ax, (3.55, 2.02), (3.30, 1.74), color=GREEN, lw=1.4)
    arrow(ax, (3.86, 2.02), (4.10, 1.74), color=GREY, lw=1.4)
    ax.text(3.70, 0.72, "A graph may be plausible yet ineligible.", ha="center", fontsize=8.2, color=DARK)

    # Card 3: locked selector output.
    box(ax, (5.48, 2.36), 1.56, 0.55, "white", edgecolor=BLUE, radius=0.08)
    box(ax, (5.48, 1.61), 1.56, 0.55, "white", edgecolor=ORANGE, radius=0.08)
    ax.text(6.26, 2.64, "Graph-lasso ranking", ha="center", va="center", fontsize=8.7)
    ax.text(6.26, 1.89, "Random-forest ranking", ha="center", va="center", fontsize=8.7)
    box(ax, (5.92, 0.83), 0.70, 0.43, "white", edgecolor=DARK, radius=0.08)
    ax.text(6.27, 1.045, "rank fusion", ha="center", va="center", fontsize=8.5)
    arrow(ax, (6.26, 2.33), (6.26, 2.19), color=BLUE, lw=1.2)
    arrow(ax, (6.26, 1.58), (6.26, 1.29), color=ORANGE, lw=1.2)
    draw_gene_list(ax, 6.84, 1.23)
    ax.text(
        6.26,
        0.57,
        "Freeze Top-20 genes,\ndirections and amplitudes",
        ha="center",
        va="center",
        fontsize=7.9,
        color=DARK,
    )

    # Card 4: untouched external scoring and the defensible conclusion.
    box(ax, (7.96, 2.53), 1.60, 0.45, "white", edgecolor=SKY, radius=0.08)
    box(ax, (7.96, 1.80), 1.60, 0.45, "white", edgecolor=BLUE, radius=0.08)
    box(ax, (7.96, 1.07), 1.60, 0.45, "white", edgecolor=GREEN, radius=0.08)
    ax.text(8.76, 2.755, "Cohort-wise scaling\n(no outcomes)", ha="center", va="center", fontsize=8.6)
    ax.text(8.76, 2.025, "Apply frozen Top-20 score", ha="center", va="center", fontsize=8.6)
    ax.text(8.76, 1.295, "C-index + stability audit", ha="center", va="center", fontsize=8.6)
    arrow(ax, (8.76, 2.50), (8.76, 2.28), color=DARK, lw=1.2)
    arrow(ax, (8.76, 1.77), (8.76, 1.55), color=DARK, lw=1.2)
    ax.text(
        8.76,
        0.66,
        "Auditable selection under\nheterogeneous graph reliability",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=DARK,
    )

    # Main left-to-right workflow arrows.
    for start, end in [((2.25, 2.00), (2.58, 2.00)), ((4.82, 2.00), (5.16, 2.00)), ((7.32, 2.00), (7.66, 2.00))]:
        arrow(ax, start, end, color=DARK, lw=2.1, mutation=15)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".svg"))
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=300)
    plt.close(fig)
    try:
        from PIL import Image

        with Image.open(output_base.with_suffix(".png")) as raster:
            raster.save(
                output_base.with_suffix(".tiff"),
                compression="tiff_lzw",
                dpi=(300, 300),
            )
    except ImportError:  # pragma: no cover - retain matplotlib TIFF fallback
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("Graphical_Abstract"),
        help="Output path without extension.",
    )
    args = parser.parse_args()
    make_figure(args.output_base)


if __name__ == "__main__":
    main()
