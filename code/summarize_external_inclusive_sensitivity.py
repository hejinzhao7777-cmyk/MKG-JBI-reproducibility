"""Create transparent five-cohort and KIRC-inclusive external summaries."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
PRIMARY_FIVE = ["LUAD", "LIHC", "COAD", "STAD", "HNSC"]
COHORTS = {
    "LUAD": "GSE31210",
    "LIHC": "GSE14520",
    "KIRC": "GSE29609",
    "COAD": "GSE39582",
    "STAD": "GSE84437",
    "HNSC": "GSE65858",
}
METHODS = {
    "MKG": "GR-SAFS_v2",
    "MKG equal-weight": "v2_equal",
    "Single-graph backend": "GR-SAFS_v1",
    "Cox-Lasso": "Cox-Lasso",
    "Cox-EN": "Cox-EN",
    "Uni-Cox": "Uni-Cox",
    "RSF": "RSF",
    "DeepSurv": "DeepSurv",
}


def exact_cluster_interval(values: np.ndarray) -> tuple[float, float]:
    size = len(values)
    means = np.empty(size**size, dtype=float)
    for position, indices in enumerate(itertools.product(range(size), repeat=size)):
        means[position] = values[list(indices)].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_rows(result_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for cancer in CANCERS:
        payload = json.loads(
            (result_root / f"final_config_comparison_{cancer}.json").read_text(encoding="utf-8")
        )[cancer]
        cohort = COHORTS[cancer]
        for label, key in METHODS.items():
            value = payload["external"][cohort][key]
            rows.append(
                {
                    "Cancer": cancer,
                    "Cohort": cohort,
                    "Analysis role": "small platform-mismatch sensitivity"
                    if cancer == "KIRC"
                    else "primary five-cohort summary",
                    "Method": label,
                    "C-index": float(value["c_index"]),
                    "Matched genes": int(value["n_matched"]),
                    "Signature size": 20,
                    "Matched fraction": float(value["n_matched"] / 20.0),
                }
            )
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame, cancers: list[str], label: str) -> pd.DataFrame:
    rows: list[dict] = []
    frame = raw[raw["Cancer"].isin(cancers)]
    for method, group in frame.groupby("Method", sort=False):
        ordered = group.set_index("Cancer").loc[cancers, "C-index"].to_numpy(dtype=float)
        low, high = exact_cluster_interval(ordered)
        rows.append(
            {
                "Analysis": label,
                "Cohorts": len(cancers),
                "Method": method,
                "Mean C-index": float(ordered.mean()),
                "Exact cohort-bootstrap low": low,
                "Exact cohort-bootstrap high": high,
            }
        )
    return pd.DataFrame(rows)


def paired_contrast(raw: pd.DataFrame, cancers: list[str], label: str) -> dict:
    pivot = raw[raw["Cancer"].isin(cancers)].pivot(index="Cancer", columns="Method", values="C-index")
    difference = (pivot.loc[cancers, "MKG"] - pivot.loc[cancers, "Uni-Cox"]).to_numpy(dtype=float)
    low, high = exact_cluster_interval(difference)
    return {
        "analysis": label,
        "cancers": cancers,
        "contrast": "MKG minus Uni-Cox",
        "mean_difference": float(difference.mean()),
        "exact_cohort_bootstrap_95pct_ci": [low, high],
        "MKG_higher_cohorts": int((difference > 0).sum()),
        "MKG_lower_cohorts": int((difference < 0).sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--cv-cox",
        type=Path,
        help="Optional development-CV-tuned Cox baseline table replacing legacy fixed penalties.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_rows(args.result_root)
    if args.cv_cox:
        cv = pd.read_csv(args.cv_cox)
        cv = cv[cv["Method"].isin(["CV-Cox-Lasso", "CV-Cox-EN"])].copy()
        cv["Analysis role"] = np.where(
            cv["Cancer"].eq("KIRC"),
            "small platform-mismatch sensitivity",
            "primary five-cohort summary",
        )
        cv["Signature size"] = 20
        cv["Matched fraction"] = cv["Matched genes"] / 20.0
        cv = cv.rename(
            columns={"External cohort": "Cohort", "External C-index": "C-index"}
        )
        raw = raw[~raw["Method"].isin(["Cox-Lasso", "Cox-EN"])]
        raw = pd.concat(
            [
                raw,
                cv[
                    [
                        "Cancer", "Cohort", "Analysis role", "Method", "C-index",
                        "Matched genes", "Signature size", "Matched fraction",
                    ]
                ],
            ],
            ignore_index=True,
        )
    five = summarize(raw, PRIMARY_FIVE, "five-cohort primary summary")
    six = summarize(raw, CANCERS, "six-cohort KIRC-inclusive sensitivity")
    summary = pd.concat([five, six], ignore_index=True)
    contrasts = [
        paired_contrast(raw, PRIMARY_FIVE, "five-cohort primary summary"),
        paired_contrast(raw, CANCERS, "six-cohort KIRC-inclusive sensitivity"),
    ]
    raw.to_csv(args.output_dir / "CMPB_EXTERNAL_COHORT_LEVEL_WITH_MATCH_COUNTS.csv", index=False)
    summary.to_csv(args.output_dir / "CMPB_EXTERNAL_PRIMARY_AND_INCLUSIVE_SUMMARY.csv", index=False)
    (args.output_dir / "CMPB_EXTERNAL_PRIMARY_AND_INCLUSIVE_CONTRASTS.json").write_text(
        json.dumps(contrasts, indent=2), encoding="utf-8"
    )
    print(raw[raw["Method"] == "MKG"].to_string(index=False))
    print(summary.to_string(index=False))
    print(json.dumps(contrasts, indent=2))


if __name__ == "__main__":
    main()
