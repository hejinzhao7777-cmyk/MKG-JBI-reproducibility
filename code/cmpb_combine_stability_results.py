"""Assemble the six-method stability source table from audited estimands."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHODS = ["MKG", "Uni-Cox", "CGBoost", "CV-Cox-EN", "CV-Cox-Lasso", "RSF"]
CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditional", type=Path, required=True)
    parser.add_argument("--tuned-cox", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conditional = pd.read_csv(args.conditional)
    conditional = conditional[conditional.Method.isin(["MKG", "Uni-Cox", "CGBoost", "RSF"])]
    tuned = pd.read_csv(args.tuned_cox)
    combined = pd.concat([conditional, tuned], ignore_index=True, sort=False)
    combined = combined[["Cancer", "Method", "normalized_RBO20", "Jaccard"]]
    combined["Cancer"] = pd.Categorical(combined["Cancer"], CANCERS, ordered=True)
    combined["Method"] = pd.Categorical(combined["Method"], METHODS, ordered=True)
    combined = combined.sort_values(["Cancer", "Method"])
    if len(combined) != len(CANCERS) * len(METHODS) or combined[["Cancer", "Method"]].duplicated().any():
        raise ValueError("stability source table is incomplete or duplicated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
