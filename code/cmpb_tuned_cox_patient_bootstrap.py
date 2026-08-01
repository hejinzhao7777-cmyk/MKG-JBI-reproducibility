"""Patient-level bootstrap intervals for CV-tuned frozen Cox scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

import final_config_comparison as F
from cmpb_cv_tuned_cox_baselines import CANCERS, COHORTS


def load_external(cancer: str, signature: dict):
    directory = F.ROOT / cancer / "external" / COHORTS[cancer]
    expression = pd.read_csv(directory / "expr.tsv", sep="\t", index_col=0)
    expression.columns = [str(column) for column in expression.columns]
    survival = pd.read_csv(directory / "survival.tsv", sep="\t")
    survival.columns = [column.lower().replace(".", "_") for column in survival.columns]
    event_column = next(column for column in survival.columns if column in {"os", "event", "status", "os_event"})
    time_column = next(column for column in survival.columns if column in {"os_time", "time", "survival_time"})
    survival = survival.set_index(survival.columns[0])
    common = sorted(set(expression.index) & set(survival.index))
    event = pd.to_numeric(survival.loc[common, event_column], errors="coerce").to_numpy()
    time = pd.to_numeric(survival.loc[common, time_column], errors="coerce").to_numpy()
    usable = np.isfinite(event) & np.isfinite(time) & (time > 0)
    risk, matched = F.frozen_risk(
        expression.loc[np.asarray(common)[usable]], signature["genes"], signature["directions"], signature["scores"]
    )
    return risk, time[usable], event[usable].astype(int), matched


def bootstrap(risk, time, event, repetitions: int, seed: int):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        index = rng.integers(0, len(time), len(time))
        try:
            values.append(concordance_index(time[index], -risk[index], event[index]))
        except ZeroDivisionError:
            continue
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cancer_index, cancer in enumerate(CANCERS):
        payload = json.loads((args.signature_dir / f"{cancer}_CV_TUNED_COX_BASELINES.json").read_text(encoding="utf-8"))
        for method_index, (method, result) in enumerate(payload.items()):
            risk, time, event, matched = load_external(cancer, result["signature"])
            point = concordance_index(time, -risk, event)
            low, high, valid = bootstrap(
                risk, time, event, args.repetitions,
                20260801 + 1000 * cancer_index + method_index,
            )
            rows.append({
                "Cancer": cancer, "Cohort": COHORTS[cancer], "Method": method,
                "n": len(time), "Events": int(event.sum()), "Matched genes": matched,
                "C-index": point, "Bootstrap low": low, "Bootstrap high": high,
                "Valid resamples": valid,
            })
    pd.DataFrame(rows).to_csv(args.output_dir / "CMPB_CV_TUNED_COX_PATIENT_BOOTSTRAP.csv", index=False)


if __name__ == "__main__":
    main()
