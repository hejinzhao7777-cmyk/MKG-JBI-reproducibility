"""Create the revised external-validation figure and its source table.

Conclusion: MKG is competitive with Uni-Cox across the five primary cohorts,
but neither is consistently superior; including the small KIRC cohort changes
the paired mean direction.  Every cancer-level point is displayed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
PRIMARY = ["LUAD", "LIHC", "COAD", "STAD", "HNSC"]
COHORTS = {"LUAD": "GSE31210", "LIHC": "GSE14520", "KIRC": "GSE29609", "COAD": "GSE39582", "STAD": "GSE84437", "HNSC": "GSE65858"}
METHODS = ["MKG", "Uni-Cox", "CV-Cox-EN", "CV-Cox-Lasso", "RSF", "DeepSurv"]
OLD_KEYS = {"MKG": "GR-SAFS_v2", "Uni-Cox": "Uni-Cox", "RSF": "RSF", "DeepSurv": "DeepSurv"}
COLORS = {"MKG": "#0072B2", "Uni-Cox": "#D55E00", "CV-Cox-EN": "#009E73", "CV-Cox-Lasso": "#E69F00", "RSF": "#6F6F6F", "DeepSurv": "#A0A0A0"}
MARKERS = {"LUAD": "o", "LIHC": "s", "KIRC": "^", "COAD": "D", "STAD": "P", "HNSC": "X"}


def source_table(result_root: Path, tuned_csv: Path) -> pd.DataFrame:
    tuned = pd.read_csv(tuned_csv).set_index(["Cancer", "Method"])
    rows = []
    for cancer in CANCERS:
        payload = json.loads((result_root / f"final_config_comparison_{cancer}.json").read_text(encoding="utf-8"))[cancer]
        cohort = COHORTS[cancer]
        for method in METHODS:
            if method in OLD_KEYS:
                value = payload["external"][cohort][OLD_KEYS[method]]
                cindex, matched = value["c_index"], value["n_matched"]
            else:
                value = tuned.loc[(cancer, method)]
                cindex, matched = value["External C-index"], value["Matched genes"]
            rows.append({"Cancer": cancer, "Cohort": cohort, "Method": method, "C-index": float(cindex), "Matched genes": int(matched), "Primary-five": cancer in PRIMARY})
    return pd.DataFrame(rows)


def plot(data: pd.DataFrame, output_dir: Path):
    mpl.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.25), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1.0]})

    x = np.arange(len(METHODS))
    offsets = np.linspace(-0.18, 0.18, len(PRIMARY))
    for offset, cancer in zip(offsets, PRIMARY):
        frame = data[data.Cancer.eq(cancer)].set_index("Method").loc[METHODS]
        axes[0].scatter(x + offset, frame["C-index"], s=28, marker=MARKERS[cancer], color=[COLORS[m] for m in METHODS], edgecolor="white", linewidth=0.4, label=cancer, zorder=3)
    means = data[data.Cancer.isin(PRIMARY)].groupby("Method")["C-index"].mean().reindex(METHODS)
    axes[0].scatter(x, means, marker="d", s=44, color="black", edgecolor="white", linewidth=0.5, label="Mean", zorder=4)
    for xpos, value in zip(x, means):
        axes[0].text(xpos, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=6.3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(METHODS, rotation=25, ha="right")
    axes[0].set_ylabel("External C-index")
    axes[0].set_ylim(0.40, 0.72)
    axes[0].axhline(0.5, color="#888888", ls=":", lw=0.7)
    axes[0].grid(axis="y", color="#D9D9D9", ls="--", lw=0.45)
    axes[0].set_title("A. Five-cohort primary transfer", loc="left")

    y = np.arange(len(CANCERS))
    pivot = data[data.Method.isin(["MKG", "Uni-Cox"])].pivot(index="Cancer", columns="Method", values="C-index").loc[CANCERS]
    for index, cancer in enumerate(CANCERS):
        axes[1].plot([pivot.loc[cancer, "MKG"], pivot.loc[cancer, "Uni-Cox"]], [index, index], color="#B8B8B8", lw=1.0, zorder=1)
    axes[1].scatter(pivot["MKG"], y, color=COLORS["MKG"], marker="o", s=34, edgecolor="white", linewidth=0.4, label="MKG", zorder=3)
    axes[1].scatter(pivot["Uni-Cox"], y, color=COLORS["Uni-Cox"], marker="s", s=32, edgecolor="white", linewidth=0.4, label="Uni-Cox", zorder=3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([f"{c}\n{COHORTS[c]}" for c in CANCERS])
    axes[1].invert_yaxis()
    axes[1].axvline(0.5, color="#888888", ls=":", lw=0.7)
    axes[1].set_xlim(0.24, 0.71)
    axes[1].set_xlabel("External C-index")
    axes[1].set_title("B. MKG--Uni-Cox paired cohorts", loc="left")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].axhspan(1.5, 2.5, color="#F2F2F2", zorder=0)
    axes[1].text(0.705, 2, "small KIRC\nsensitivity", ha="right", va="center", fontsize=6.2, color="#555555")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=6, frameon=False, handletextpad=0.3, columnspacing=0.8)
    for extension, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 600}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(output_dir / f"Fig3_generalization.{extension}", **kwargs)
    fig.savefig(output_dir / "Fig3_generalization_preview.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--tuned-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = source_table(args.result_root, args.tuned_csv)
    data.to_csv(args.output_dir / "Fig3_generalization_source.csv", index=False)
    plot(data, args.output_dir)


if __name__ == "__main__":
    main()
