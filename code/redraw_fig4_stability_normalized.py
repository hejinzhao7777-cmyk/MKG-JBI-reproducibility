from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
FIG = ROOT.parent / "07_论文初稿" / "figures"
TAB = ROOT.parent / "07_论文初稿" / "tables" / "Table1_stability.csv"
DENOM = 1 - 0.9 ** 20


def main():
    df = pd.read_csv(TAB)
    df = df[df["Cancer"] != "Mean"].copy()
    if "MKG RBO" in df.columns:
        df["MKG normalized RBO@20"] = df["MKG RBO"] / DENOM
        df["Cox-Lasso normalized RBO@20"] = df["Cox-Lasso RBO"] / DENOM
    x = np.arange(len(df))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    axes[0].bar(x - width / 2, df["MKG normalized RBO@20"], width, label="MKG", color="#2E7D32")
    axes[0].bar(x + width / 2, df["Cox-Lasso normalized RBO@20"], width, label="Cox-Lasso", color="#E45756")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["Cancer"])
    axes[0].set_ylabel("Normalized RBO@20")
    axes[0].set_title("A. Rank-sensitive stability")
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, df["MKG Jaccard"], width, label="MKG", color="#2E7D32")
    axes[1].bar(x + width / 2, df["Cox-Lasso Jaccard"], width, label="Cox-Lasso", color="#E45756")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["Cancer"])
    axes[1].set_ylabel("Jaccard")
    axes[1].set_title("B. Set-level stability")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "Fig4_stability.pdf")
    fig.savefig(FIG / "Fig4_stability.png", dpi=300)


if __name__ == "__main__":
    main()

