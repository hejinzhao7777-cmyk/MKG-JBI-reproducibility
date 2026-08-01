"""Assemble the final eight-cohort external C-index source table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked-ci", type=Path, required=True)
    parser.add_argument("--cv-ci", type=Path, required=True)
    parser.add_argument("--additional", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    locked = pd.read_csv(args.locked_ci)
    locked = locked[locked["Method"].isin(["MKG", "Uni-Cox", "RSF", "DeepSurv"])].copy()
    locked["Method"] = locked["Method"].replace(
        {"RSF": "RSF ranking", "DeepSurv": "DeepSurv attribution"}
    )
    locked = locked.rename(
        columns={
            "External cohort": "Cohort",
            "Matched Top-20 genes": "Matched genes",
            "CI low": "95% CI low",
            "CI high": "95% CI high",
            "Valid bootstrap replicates": "Valid resamples",
        }
    )

    cv = pd.read_csv(args.cv_ci).rename(
        columns={"Bootstrap low": "95% CI low", "Bootstrap high": "95% CI high"}
    )
    roles = locked[["Cancer", "Cohort", "Role"]].drop_duplicates()
    cv = cv.merge(roles, on=["Cancer", "Cohort"], how="left", validate="many_to_one")

    additional = pd.read_csv(args.additional).rename(
        columns={"Bootstrap low": "95% CI low", "Bootstrap high": "95% CI high"}
    )
    additional["Role"] = "Additional within-cancer replication"

    columns = [
        "Cancer", "Cohort", "Role", "Method", "n", "Events", "Matched genes",
        "C-index", "95% CI low", "95% CI high", "Valid resamples",
    ]
    combined = pd.concat(
        [locked[columns], cv[columns], additional[columns]], ignore_index=True
    )
    method_order = {
        "MKG": 0, "Uni-Cox": 1, "CV-Cox-Lasso": 2, "CV-Cox-EN": 3,
        "RSF ranking": 4, "DeepSurv attribution": 5,
    }
    cohort_order = {
        "GSE31210": 0, "GSE50081": 1, "GSE14520": 2, "GSE76427": 3,
        "GSE29609": 4, "GSE39582": 5, "GSE84437": 6, "GSE65858": 7,
    }
    combined["_cohort"] = combined["Cohort"].map(cohort_order)
    combined["_method"] = combined["Method"].map(method_order)
    if combined[["_cohort", "_method"]].isna().any().any():
        raise RuntimeError("Unexpected cohort or method while assembling external table")
    combined = combined.sort_values(["_cohort", "_method"]).drop(columns=["_cohort", "_method"])
    if len(combined) != 48 or combined.duplicated(["Cohort", "Method"]).any():
        raise RuntimeError("Expected 8 cohorts x 6 unique methods")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined)} rows to {args.output}")


if __name__ == "__main__":
    main()
