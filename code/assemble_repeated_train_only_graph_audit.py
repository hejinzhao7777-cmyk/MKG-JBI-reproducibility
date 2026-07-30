"""Assemble repeated train-only graph-reconstruction audits.

The unit of repetition is a prespecified stratified train/test split.  Primary
inference is cancer-clustered: split results are first averaged within each
cancer, and the six cancer means are then bootstrapped.  This avoids treating
the 18 cancer-by-split rows as independent biological cohorts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CANCER_ORDER = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
SPLIT_SEEDS = [42, 2025, 7301]
GATE_MARGINS = [0.0, 0.005, 0.010]
EXPECTED_CONFIGURATION = {
    "bootstrap_B": 10,
    "stage1_rf_trees": 100,
    "stage1_pgd_iterations": 300,
    "stability_pgd_iterations": 300,
    "rf_jobs": 1,
    "layer_workers": 1,
    "top_k": 20,
}
CANCER_STYLE = {
    "LUAD": ("#0072B2", "o"),
    "LIHC": ("#E69F00", "s"),
    "KIRC": ("#009E73", "^"),
    "COAD": ("#CC79A7", "D"),
    "STAD": ("#56B4E9", "P"),
    "HNSC": ("#D55E00", "X"),
}


def load_results(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for path in sorted(root.rglob("CMPB_REPEATED_TRAIN_ONLY_GRAPH_AUDIT.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (result["cancer"], int(result["split_seed"]))
        if key in seen:
            raise ValueError(f"duplicate cancer/seed result: {key} ({path})")
        if key[0] not in CANCER_ORDER or key[1] not in SPLIT_SEEDS:
            raise ValueError(f"unexpected cancer/seed result: {key} ({path})")
        seen.add(key)
        observed_margins = sorted(
            float(value) for value in result["reconstructed"]["margins"]
        )
        if not np.allclose(observed_margins, GATE_MARGINS, rtol=0, atol=1e-12):
            raise ValueError(
                f"unexpected gate margins for {key}: {observed_margins}"
            )
        fixed_margins = sorted(float(value) for value in result["fixed"]["margins"])
        if not np.allclose(fixed_margins, GATE_MARGINS, rtol=0, atol=1e-12):
            raise ValueError(
                f"unexpected fixed-graph gate margins for {key}: {fixed_margins}"
            )
        for field, expected_value in EXPECTED_CONFIGURATION.items():
            observed_value = result["configuration"].get(field)
            if observed_value != expected_value:
                raise ValueError(
                    f"configuration mismatch for {key}: {field}="
                    f"{observed_value!r}, expected {expected_value!r}"
                )
        if int(result["algorithm_seed"]) != 42:
            raise ValueError(
                f"algorithm seed mismatch for {key}: {result['algorithm_seed']}"
            )
        if not np.isclose(float(result["test_size"]), 0.30):
            raise ValueError(
                f"test-size mismatch for {key}: {result['test_size']}"
            )
        for margin_key, reconstructed in result["reconstructed"]["margins"].items():
            fixed = result["fixed"]["margins"][margin_key]
            rows.append(
                {
                    "cancer": result["cancer"],
                    "split_seed": int(result["split_seed"]),
                    "train_n": int(result["train_n"]),
                    "test_n": int(result["test_n"]),
                    "test_events": int(result["test_events"]),
                    "gate_margin": float(margin_key),
                    "no_graph_cindex": float(result["baseline"]["test_cindex"]),
                    "reconstructed_cindex": float(reconstructed["test_cindex"]),
                    "fixed_cindex": float(fixed["test_cindex"]),
                    "fixed_minus_reconstructed": float(
                        reconstructed["fixed_minus_reconstructed_cindex"]
                    ),
                    "top20_overlap": int(reconstructed["fixed_overlap_count"]),
                    "top20_jaccard": float(reconstructed["fixed_jaccard"]),
                    "reconstructed_mode": reconstructed["mode"],
                    "fixed_mode": fixed["mode"],
                    "route_agreement": reconstructed["mode"] == fixed["mode"],
                    "reconstructed_reject": (
                        reconstructed["mode"] == "reject_all_graphs"
                    ),
                    "fixed_reject": fixed["mode"] == "reject_all_graphs",
                    "runtime_seconds": float(result["runtime_seconds"]),
                    "source_json": path.relative_to(root).as_posix(),
                }
            )
    if not rows:
        raise FileNotFoundError(f"no audit JSON files found under {root}")
    expected = {(cancer, seed) for cancer in CANCER_ORDER for seed in SPLIT_SEEDS}
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(
            "incomplete repeated audit; missing cancer/seed pairs: "
            + ", ".join(f"{cancer}/{seed}" for cancer, seed in missing)
        )
    raw = pd.DataFrame(rows)
    raw["cancer"] = pd.Categorical(
        raw["cancer"], categories=CANCER_ORDER, ordered=True
    )
    return raw.sort_values(["gate_margin", "cancer", "split_seed"]).reset_index(
        drop=True
    )


def clustered_interval(
    cancer_means: np.ndarray,
    repetitions: int = 50_000,
    seed: int = 20260730,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(cancer_means)
    sampled = cancer_means[rng.integers(0, n, size=(repetitions, n))].mean(axis=1)
    return tuple(np.quantile(sampled, [0.025, 0.975]).tolist())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summaries(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    by_cancer = (
        raw.groupby(["gate_margin", "cancer"], observed=True)
        .agg(
            split_n=("split_seed", "nunique"),
            reconstructed_cindex_mean=("reconstructed_cindex", "mean"),
            fixed_cindex_mean=("fixed_cindex", "mean"),
            fixed_minus_reconstructed_mean=("fixed_minus_reconstructed", "mean"),
            fixed_minus_reconstructed_min=("fixed_minus_reconstructed", "min"),
            fixed_minus_reconstructed_max=("fixed_minus_reconstructed", "max"),
            top20_jaccard_mean=("top20_jaccard", "mean"),
            route_agreement_rate=("route_agreement", "mean"),
            reconstructed_reject_rate=("reconstructed_reject", "mean"),
            fixed_reject_rate=("fixed_reject", "mean"),
        )
        .reset_index()
    )

    aggregate_rows = []
    interval_manifest = {}
    for margin, sub in by_cancer.groupby("gate_margin", observed=True):
        values = sub["fixed_minus_reconstructed_mean"].to_numpy(dtype=float)
        low, high = clustered_interval(values)
        raw_sub = raw[raw["gate_margin"] == margin]
        aggregate_rows.append(
            {
                "gate_margin": float(margin),
                "cancers": int(sub["cancer"].nunique()),
                "splits_per_cancer_min": int(sub["split_n"].min()),
                "split_rows": int(len(raw_sub)),
                "reconstructed_cindex_mean": float(
                    sub["reconstructed_cindex_mean"].mean()
                ),
                "fixed_cindex_mean": float(sub["fixed_cindex_mean"].mean()),
                "fixed_minus_reconstructed_mean": float(values.mean()),
                "cancer_clustered_ci_low": low,
                "cancer_clustered_ci_high": high,
                "fixed_minus_reconstructed_median": float(np.median(values)),
                "top20_jaccard_mean": float(sub["top20_jaccard_mean"].mean()),
                "top20_jaccard_range_min": float(raw_sub["top20_jaccard"].min()),
                "top20_jaccard_range_max": float(raw_sub["top20_jaccard"].max()),
                "route_agreement_rate": float(raw_sub["route_agreement"].mean()),
                "reconstructed_reject_rate": float(
                    raw_sub["reconstructed_reject"].mean()
                ),
                "fixed_reject_rate": float(raw_sub["fixed_reject"].mean()),
            }
        )
        interval_manifest[f"{margin:.3f}"] = {
            "estimand": "mean of six cancer-specific split means",
            "bootstrap_unit": "cancer",
            "bootstrap_repetitions": 50_000,
            "seed": 20260730,
            "ci": [low, high],
        }
    return by_cancer, pd.DataFrame(aggregate_rows), interval_manifest


def make_figure(raw: pd.DataFrame, output: Path) -> None:
    primary = raw[raw["gate_margin"] == 0.0].copy()
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(7.48, 2.75), constrained_layout=True
    )

    for index, cancer in enumerate(CANCER_ORDER):
        sub = primary[primary["cancer"] == cancer].sort_values("split_seed")
        x = np.full(len(sub), index, dtype=float) + np.linspace(-0.16, 0.16, len(sub))
        color, marker = CANCER_STYLE[cancer]
        axes[0].scatter(
            x,
            sub["fixed_minus_reconstructed"],
            s=25,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        axes[1].scatter(
            x,
            sub["top20_jaccard"],
            s=25,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        for axis, column in (
            (axes[0], "fixed_minus_reconstructed"),
            (axes[1], "top20_jaccard"),
        ):
            mean = float(sub[column].mean())
            axis.plot(
                [index - 0.22, index + 0.22],
                [mean, mean],
                color="black",
                linewidth=1.1,
                solid_capstyle="round",
                zorder=4,
            )

    axes[0].axhline(0, color="0.35", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Fixed minus reconstructed\nheld-out C-index")
    axes[1].set_ylabel("Top-20 Jaccard")
    axes[1].set_ylim(0, 1)
    for axis in axes[:2]:
        axis.set_xticks(range(len(CANCER_ORDER)), CANCER_ORDER, rotation=32)
        axis.tick_params(axis="x", pad=1)
        axis.grid(axis="y", color="0.88", linewidth=0.45, linestyle="--")

    margins = sorted(raw["gate_margin"].unique())
    route_agreement = [
        raw.loc[raw["gate_margin"] == margin, "route_agreement"].mean()
        for margin in margins
    ]
    reconstructed_reject = [
        raw.loc[raw["gate_margin"] == margin, "reconstructed_reject"].mean()
        for margin in margins
    ]
    x = np.asarray(margins, dtype=float)
    axes[2].plot(
        x,
        route_agreement,
        label="Route agreement",
        color="#4C78A8",
        marker="o",
        linewidth=1.2,
        markersize=4.2,
    )
    axes[2].plot(
        x,
        reconstructed_reject,
        label="Train-only rejection",
        color="#F58518",
        marker="s",
        linestyle="--",
        linewidth=1.2,
        markersize=4.2,
    )
    axes[2].set_xticks(x, [f"{margin:.3f}" for margin in margins])
    axes[2].set_xlabel("Minimum gain margin")
    axes[2].set_ylabel("Fraction of 18 audits")
    axes[2].set_ylim(0, 1)
    axes[2].legend(frameon=False, loc="best")
    axes[2].grid(axis="y", color="0.88", linewidth=0.45, linestyle="--")

    for label, axis in zip(("(A)", "(B)", "(C)"), axes):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            -0.19,
            1.03,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8.5,
        )
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = load_results(root)
    by_cancer, aggregate, intervals = summaries(raw)

    raw.to_csv(outdir / "CMPB_REPEATED_TRAIN_ONLY_RAW.csv", index=False)
    by_cancer.to_csv(
        outdir / "CMPB_REPEATED_TRAIN_ONLY_BY_CANCER.csv", index=False
    )
    aggregate.to_csv(
        outdir / "CMPB_REPEATED_TRAIN_ONLY_AGGREGATE.csv", index=False
    )
    manifest = {
        "source_root": root.name,
        "source_json_sha256": {
            path.relative_to(root).as_posix(): sha256(path)
            for path in sorted(
                root.rglob("CMPB_REPEATED_TRAIN_ONLY_GRAPH_AUDIT.json")
            )
        },
        "cancers_expected": CANCER_ORDER,
        "split_seeds": sorted(raw["split_seed"].unique().tolist()),
        "gate_margins": sorted(raw["gate_margin"].unique().tolist()),
        "n_unique_cancer_seed_pairs": int(
            raw[["cancer", "split_seed"]].drop_duplicates().shape[0]
        ),
        "intervals": intervals,
        "interpretation": (
            "A sensitivity audit across prespecified data splits, not repeated "
            "nested cross-validation or independent cohort replication."
        ),
    }
    (outdir / "CMPB_REPEATED_TRAIN_ONLY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    make_figure(raw, outdir / "Fig_CMPB_repeated_train_only_audit")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
