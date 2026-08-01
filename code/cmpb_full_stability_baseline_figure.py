#!/usr/bin/env python
"""Create the complete six-method stability comparison used in the CMPB paper.

The six cancer cohorts are the independent display/analysis units.  Because
n=6 per method, every cohort value is shown directly; no box or violin
summary is used.  Paired MKG contrasts use the exact empirical
percentile distribution over all 6^6 cancer-cluster bootstrap resamples.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


try:
    from visual_qa import audit_layout, print_report, render_preview
except ImportError:  # pragma: no cover - optional local QA helper
    audit_layout = print_report = render_preview = None


METHOD_ORDER = ["MKG", "Uni-Cox", "CGBoost", "CV-Cox-EN", "CV-Cox-Lasso", "RSF"]
CANCER_ORDER = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
METRICS = [
    ("normalized_RBO20", "Normalized RBO@20", "(A)"),
    ("Jaccard", "Top-20 Jaccard", "(B)"),
]

# Okabe--Ito colors plus redundant marker coding for grayscale accessibility.
CANCER_STYLE = {
    "LUAD": ("#0072B2", "o"),
    "LIHC": ("#E69F00", "s"),
    "KIRC": ("#009E73", "^"),
    "COAD": ("#CC79A7", "D"),
    "STAD": ("#56B4E9", "P"),
    "HNSC": ("#D55E00", "X"),
}


def exact_cluster_bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    """Exact percentile CI over all cluster bootstrap resamples."""
    n = values.size
    indices = np.fromiter(
        (i for draw in itertools.product(range(n), repeat=n) for i in draw),
        dtype=np.int16,
        count=(n**n) * n,
    ).reshape(-1, n)
    means = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def analyse(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    summary = (
        df.groupby("Method", sort=False)[["normalized_RBO20", "Jaccard"]]
        .agg(["mean", "median", "std"])
        .reindex(METHOD_ORDER)
    )
    rows: list[dict] = []
    paired: dict[str, dict] = {}
    for metric, label, _ in METRICS:
        pivot = df.pivot(index="Cancer", columns="Method", values=metric).reindex(
            CANCER_ORDER
        )
        paired[metric] = {"label": label, "comparisons": {}}
        for comparator in ("Uni-Cox", "CV-Cox-EN"):
            diff = (pivot["MKG"] - pivot[comparator]).to_numpy(float)
            lo, hi = exact_cluster_bootstrap_ci(diff)
            test = wilcoxon(diff, alternative="two-sided", method="exact")
            paired[metric]["comparisons"][f"MKG - {comparator}"] = {
                "n_cancers": int(diff.size),
                "mean_difference": float(diff.mean()),
                "median_difference": float(np.median(diff)),
                "exact_cancer_cluster_bootstrap_95pct_ci": [lo, hi],
                "mkg_higher_cancers": int(np.sum(diff > 0)),
                "mkg_lower_cancers": int(np.sum(diff < 0)),
                "wilcoxon_exact_two_sided_p": float(test.pvalue),
                "per_cancer_differences": {
                    cancer: float(value) for cancer, value in zip(CANCER_ORDER, diff)
                },
            }
        for method in METHOD_ORDER:
            vals = pivot[method].to_numpy(float)
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "n_cancers": int(vals.size),
                    "mean": float(vals.mean()),
                    "median": float(np.median(vals)),
                    "sd_across_cancers": float(vals.std(ddof=1)),
                }
            )
    return pd.DataFrame(rows), paired


def plot(df: pd.DataFrame, outdir: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(7.48, 3.15), sharey=False, constrained_layout=True
    )
    x = np.arange(len(METHOD_ORDER), dtype=float)
    offsets = np.linspace(-0.20, 0.20, len(CANCER_ORDER))

    for ax, (metric, ylabel, panel) in zip(axes, METRICS):
        pivot = (
            df.pivot(index="Cancer", columns="Method", values=metric)
            .reindex(index=CANCER_ORDER, columns=METHOD_ORDER)
        )
        for offset, cancer in zip(offsets, CANCER_ORDER):
            color, marker = CANCER_STYLE[cancer]
            ax.scatter(
                x + offset,
                pivot.loc[cancer].to_numpy(float),
                s=28,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.92,
                zorder=3,
                label=cancer if metric == METRICS[0][0] else None,
            )

        means = pivot.mean(axis=0).to_numpy(float)
        ax.scatter(
            x,
            means,
            s=42,
            marker="d",
            facecolor="black",
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
            label="Mean" if metric == METRICS[0][0] else None,
        )
        for xpos, mean in zip(x, means):
            ax.text(
                xpos,
                mean + 0.014,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.3,
                color="black",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=24, ha="right")
        ax.set_ylabel(ylabel)
        observed_max = float(pivot.to_numpy().max())
        minimum_ceiling = 0.50 if metric == "normalized_RBO20" else 0.42
        ax.set_ylim(0.0, max(minimum_ceiling, observed_max + 0.05))
        ax.set_xlim(-0.55, len(METHOD_ORDER) - 0.45)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, linestyle="--")
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.16,
            1.03,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=7,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    preview = outdir / "Fig4_stability_preview.png"
    gray = outdir / "Fig4_stability_grayscale.png"
    if render_preview is not None:
        render_preview(fig, str(preview), dpi=180)
        issues = audit_layout(fig)
        print_report(issues)
    else:
        fig.savefig(preview, dpi=180, bbox_inches="tight", facecolor="white")

    # Preserve the exact Elsevier double-column canvas.  Constrained layout
    # already reserves space for every label and the figure-level legend.
    fig.savefig(outdir / "Fig4_stability.pdf")
    fig.savefig(outdir / "Fig4_stability.svg")
    fig.savefig(outdir / "Fig4_stability.png", dpi=600)
    fig.savefig(
        outdir / "Fig4_stability.tiff",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )

    # Grayscale is a review artifact, not a submission figure.
    import matplotlib.image as mpimg

    rgb = mpimg.imread(preview)
    luminance = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    plt.imsave(gray, luminance, cmap="gray", vmin=0, vmax=1)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"Cancer", "Method", "normalized_RBO20", "Jaccard"}
    if set(df.columns) != required:
        raise ValueError(f"expected columns {sorted(required)}, got {df.columns.tolist()}")
    if set(df["Cancer"]) != set(CANCER_ORDER):
        raise ValueError("unexpected cancer set")
    if set(df["Method"]) != set(METHOD_ORDER):
        raise ValueError("unexpected method set")
    if df.groupby("Method").size().ne(6).any():
        raise ValueError("each method must have exactly six cancer-level values")

    summary, paired = analyse(df)
    summary.to_csv(args.outdir / "CMPB_STABILITY_SIX_METHOD_SUMMARY.csv", index=False)
    (args.outdir / "CMPB_STABILITY_PAIRED_CONTRASTS.json").write_text(
        json.dumps(paired, indent=2), encoding="utf-8"
    )
    plot(df, args.outdir)
    print(json.dumps(paired, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
