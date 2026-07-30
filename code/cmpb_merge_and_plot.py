"""Merge per-cancer CMPB ablation workers, summarize uncertainty, and plot."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cmpb_five_arm_ablation as A


OUTDIR = Path(os.environ.get("MKG_CMPB_LOCK_DIR", str(A.OUTDIR)))
# Keep the imported worker's table writer on the same explicitly selected
# directory. This matters when the plotting script is run from a released
# snapshot rather than from the original analysis tree.
A.OUTDIR = OUTDIR
A.CACHE_PATH = OUTDIR / f"CMPB_FIVE_ARM_ABLATION_CACHE{A.RUN_SUFFIX}.json"
A.TABLE_PATH = OUTDIR / f"CMPB_FIVE_ARM_ABLATION{A.RUN_SUFFIX}.csv"
A.STABILITY_RAW_PATH = OUTDIR / f"CMPB_FIVE_ARM_STABILITY_RAW{A.RUN_SUFFIX}.json"
A.STABILITY_TABLE_PATH = OUTDIR / f"CMPB_FIVE_ARM_STABILITY{A.RUN_SUFFIX}.csv"
A.RESOURCE_PATH = OUTDIR / f"CMPB_COMPUTATIONAL_COST{A.RUN_SUFFIX}.csv"
FIGURE_PATH = OUTDIR / "Fig_CMPB_five_arm_ablation.pdf"
FIGURE_PNG_PATH = OUTDIR / "Fig_CMPB_five_arm_ablation.png"
SUMMARY_PATH = OUTDIR / "CMPB_FIVE_ARM_STATISTICAL_SUMMARY.json"
SCHEME_LABELS = {
    "no_graph": "No graph",
    "equal": "Equal",
    "stability_only": "Stability only",
    "utility_only": "Utility only",
    "joint": "Joint MKG",
}
SCHEME_COLORS = {
    "no_graph": "#6B6B6B",
    "equal": "#56B4E9",
    "stability_only": "#009E73",
    "utility_only": "#E69F00",
    "joint": "#D55E00",
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_workers():
    cache = {}
    stability = {}
    missing = []
    for cancer in A.CANCERS:
        cache_path = OUTDIR / f"CMPB_FIVE_ARM_ABLATION_CACHE_{cancer}.json"
        stability_path = OUTDIR / f"CMPB_FIVE_ARM_STABILITY_RAW_{cancer}.json"
        if not cache_path.exists() or not stability_path.exists():
            missing.append(cancer)
            continue
        cache.update(load(cache_path))
        stability.update(load(stability_path))
    if missing:
        raise RuntimeError("Missing completed worker outputs: " + ", ".join(missing))
    save(OUTDIR / "CMPB_FIVE_ARM_ABLATION_CACHE.json", cache)
    save(OUTDIR / "CMPB_FIVE_ARM_STABILITY_RAW.json", stability)
    A.write_tables(cache, stability)
    return cache


def bootstrap(values, n_boot=50000, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def summarize(real, stability):
    primary_real = real[real["Cancer"].isin(A.PRIMARY_CANCERS)]
    primary_stability = stability[stability["Cancer"].isin(A.PRIMARY_CANCERS)]
    summary = {"external_cindex": {}, "normalized_rbo": {}, "paired_differences": {}}
    for scheme in A.SCHEMES:
        values = primary_real.loc[
            primary_real["Scheme"] == scheme, "External frozen-score C-index"
        ].to_numpy()
        rbo = primary_stability.loc[
            primary_stability["Scheme"] == scheme, "normalized_RBO@20"
        ].to_numpy()
        summary["external_cindex"][scheme] = bootstrap(values)
        summary["normalized_rbo"][scheme] = bootstrap(rbo)

    pivot_c = primary_real.pivot(
        index="Cancer", columns="Scheme", values="External frozen-score C-index"
    ).loc[A.PRIMARY_CANCERS]
    pivot_r = primary_stability.pivot(
        index="Cancer", columns="Scheme", values="normalized_RBO@20"
    ).loc[A.PRIMARY_CANCERS]
    for scheme in A.SCHEMES:
        if scheme == "joint":
            continue
        summary["paired_differences"][f"joint_minus_{scheme}_cindex"] = bootstrap(
            pivot_c["joint"] - pivot_c[scheme], seed=43
        )
        summary["paired_differences"][f"joint_minus_{scheme}_rbo"] = bootstrap(
            pivot_r["joint"] - pivot_r[scheme], seed=44
        )
    save(SUMMARY_PATH, summary)
    return summary, pivot_c, pivot_r


def plot(summary, pivot_c, pivot_r):
    synthetic = pd.read_csv(OUTDIR / "CMPB_SYNTHETIC_FIVE_ARM_SUMMARY.csv")
    real = pd.read_csv(OUTDIR / "CMPB_FIVE_ARM_ABLATION.csv")
    cancer_rows = real[real["Cancer"].isin(A.CANCERS)]

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(7.48, 5.15), constrained_layout=True
    )

    scenarios = list(synthetic["scenario"].drop_duplicates())
    matrix = synthetic.pivot(
        index="scheme", columns="scenario", values="mean_test_gain"
    ).loc[A.SCHEMES, scenarios]
    im = axes[0, 0].imshow(matrix.values, cmap="RdBu_r", vmin=-0.25, vmax=0.25)
    axes[0, 0].set_xticks(range(len(scenarios)))
    axes[0, 0].set_xticklabels(["R1", "R2", "R3", "R4", "R5"])
    axes[0, 0].set_yticks(range(len(A.SCHEMES)))
    axes[0, 0].set_yticklabels([SCHEME_LABELS[s] for s in A.SCHEMES])
    for i in range(len(A.SCHEMES)):
        for j in range(len(scenarios)):
            value = matrix.iloc[i, j]
            axes[0, 0].text(
                j,
                i,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if abs(value) > 0.14 else "black",
            )
    axes[0, 0].set_title("Independent-test gain in controlled scenarios", loc="left")
    cbar = fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.03)
    cbar.set_label("Negative-MSE gain vs no graph", fontsize=6.5)

    x = np.arange(len(A.SCHEMES))
    rbo_means = [summary["normalized_rbo"][s]["mean"] for s in A.SCHEMES]
    rbo_low = [summary["normalized_rbo"][s]["ci_low"] for s in A.SCHEMES]
    rbo_high = [summary["normalized_rbo"][s]["ci_high"] for s in A.SCHEMES]
    axes[0, 1].errorbar(
        x,
        rbo_means,
        yerr=[
            np.array(rbo_means) - np.array(rbo_low),
            np.array(rbo_high) - np.array(rbo_means),
        ],
        fmt="D",
        markersize=4.2,
        markerfacecolor="black",
        markeredgecolor="white",
        markeredgewidth=0.4,
        color="black",
        capsize=3,
        lw=1,
    )
    for i, scheme in enumerate(A.SCHEMES):
        jitter = np.linspace(-0.12, 0.12, len(pivot_r))
        axes[0, 1].scatter(
            np.full(len(pivot_r), i) + jitter,
            pivot_r[scheme],
            s=18,
            facecolor=SCHEME_COLORS[scheme],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([SCHEME_LABELS[s] for s in A.SCHEMES], rotation=25, ha="right")
    axes[0, 1].set_ylabel("Normalized RBO@20")
    axes[0, 1].set_title("Five-cohort selection stability", loc="left")

    c_means = [summary["external_cindex"][s]["mean"] for s in A.SCHEMES]
    c_low = [summary["external_cindex"][s]["ci_low"] for s in A.SCHEMES]
    c_high = [summary["external_cindex"][s]["ci_high"] for s in A.SCHEMES]
    axes[1, 0].errorbar(
        x,
        c_means,
        yerr=[
            np.array(c_means) - np.array(c_low),
            np.array(c_high) - np.array(c_means),
        ],
        fmt="D",
        markersize=4.2,
        markerfacecolor="black",
        markeredgecolor="white",
        markeredgewidth=0.4,
        color="black",
        capsize=3,
        lw=1,
    )
    for i, scheme in enumerate(A.SCHEMES):
        jitter = np.linspace(-0.12, 0.12, len(pivot_c))
        axes[1, 0].scatter(
            np.full(len(pivot_c), i) + jitter,
            pivot_c[scheme],
            s=18,
            facecolor=SCHEME_COLORS[scheme],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    axes[1, 0].axhline(0.5, color="#777777", ls="--", lw=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([SCHEME_LABELS[s] for s in A.SCHEMES], rotation=25, ha="right")
    axes[1, 0].set_ylabel("External C-index")
    axes[1, 0].set_ylim(0.45, 0.68)
    axes[1, 0].set_title("Frozen-score external discrimination", loc="left")

    utility = cancer_rows[cancer_rows["Scheme"] == "utility_only"].set_index("Cancer")
    joint = cancer_rows[cancer_rows["Scheme"] == "joint"].set_index("Cancer")
    distances = []
    for cancer in A.CANCERS:
        columns = ["w_no_relation", "w_coexpr", "w_meth", "w_cnv"]
        distances.append(float(np.abs(utility.loc[cancer, columns] - joint.loc[cancer, columns]).sum()))
    d_x = np.arange(len(A.CANCERS))
    axes[1, 1].vlines(d_x, 0, distances, color="#7A5195", linewidth=1.0)
    axes[1, 1].scatter(
        d_x,
        distances,
        color="#7A5195",
        edgecolor="white",
        linewidth=0.5,
        s=28,
        marker="D",
        zorder=3,
    )
    axes[1, 1].set_xticks(np.arange(len(A.CANCERS)))
    axes[1, 1].set_xticklabels(A.CANCERS)
    axes[1, 1].set_ylabel("L1 distance between routing weights")
    axes[1, 1].set_title("Utility-only vs joint routing decisions", loc="left")
    axes[1, 1].set_ylim(bottom=0)

    for label, axis in zip(("(A)", "(B)", "(C)", "(D)"), axes.flat):
        axis.text(
            -0.16,
            1.03,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8.5,
        )
    fig.savefig(FIGURE_PATH)
    fig.savefig(FIGURE_PNG_PATH, dpi=600)
    fig.savefig(FIGURE_PATH.with_suffix(".svg"))
    fig.savefig(FIGURE_PATH.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def main():
    merge_workers()
    real = pd.read_csv(OUTDIR / "CMPB_FIVE_ARM_ABLATION.csv")
    stability = pd.read_csv(OUTDIR / "CMPB_FIVE_ARM_STABILITY.csv")
    summary, pivot_c, pivot_r = summarize(real, stability)
    plot(summary, pivot_c, pivot_r)
    print(json.dumps(summary, indent=2))
    print(f"Saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
