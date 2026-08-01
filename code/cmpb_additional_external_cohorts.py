"""Audit every locked secondary external cohort under the frozen-score contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

import final_config_comparison as F


COHORTS = {"LUAD": ["GSE50081"], "LIHC": ["GSE76427"]}
LOCKED_METHODS = {
    "MKG": "GR-SAFS_v2",
    "Uni-Cox": "Uni-Cox",
    "RSF ranking": "RSF",
    "DeepSurv attribution": "DeepSurv",
}
TUNED_METHODS = ("CV-Cox-Lasso", "CV-Cox-EN")


def load_external(cancer: str, cohort: str, signature: dict):
    directory = F.ROOT / cancer / "external" / cohort
    expression = pd.read_csv(directory / "expr.tsv", sep="\t", index_col=0)
    expression.columns = [str(column) for column in expression.columns]
    survival = pd.read_csv(directory / "survival.tsv", sep="\t")
    survival.columns = [column.lower().replace(".", "_") for column in survival.columns]
    event_column = next(column for column in survival.columns if column in {"os", "event", "status", "os_event"})
    time_column = next(column for column in survival.columns if column in {"os_time", "time", "survival_time"})
    survival = survival.set_index(survival.columns[0])
    common = np.asarray(sorted(set(expression.index) & set(survival.index)))
    event = pd.to_numeric(survival.loc[common, event_column], errors="coerce").to_numpy()
    time = pd.to_numeric(survival.loc[common, time_column], errors="coerce").to_numpy()
    usable = np.isfinite(event) & np.isfinite(time) & (time > 0)
    risk, matched = F.frozen_risk(
        expression.loc[common[usable]], signature["genes"], signature["directions"], signature["scores"]
    )
    if risk is None:
        raise RuntimeError(f"No signature genes matched {cancer} {cohort}")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-results", type=Path, required=True)
    parser.add_argument("--tuned-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cancer_index, (cancer, cohorts) in enumerate(COHORTS.items()):
        locked = json.loads((args.lock_results / f"final_config_comparison_{cancer}.json").read_text(encoding="utf-8"))[cancer]
        tuned = json.loads((args.tuned_results / f"{cancer}_CV_TUNED_COX_BASELINES.json").read_text(encoding="utf-8"))
        signatures = {label: locked["method_signatures"][key] for label, key in LOCKED_METHODS.items()}
        signatures.update({method: tuned[method]["signature"] for method in TUNED_METHODS})
        for cohort_index, cohort in enumerate(cohorts):
            for method_index, (method, signature) in enumerate(signatures.items()):
                risk, time, event, matched = load_external(cancer, cohort, signature)
                point = concordance_index(time, -risk, event)
                low, high, valid = bootstrap(
                    risk, time, event, args.repetitions,
                    20260801 + 10000 * cancer_index + 100 * cohort_index + method_index,
                )
                rows.append({
                    "Cancer": cancer, "Cohort": cohort, "Method": method,
                    "n": len(time), "Events": int(event.sum()),
                    "Median OS days": float(np.median(time)), "Matched genes": matched,
                    "C-index": point, "Bootstrap low": low, "Bootstrap high": high,
                    "Valid resamples": valid,
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "CMPB_ADDITIONAL_EXTERNAL_COHORTS.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
