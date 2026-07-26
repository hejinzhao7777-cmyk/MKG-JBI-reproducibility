"""Bootstrap confidence intervals for the locked JBI submission results.

This script never refits a signature. It reconstructs each frozen molecular
score from the submission-lock signature and the prespecified external cohort,
then bootstraps patients to estimate C-index uncertainty. Stability uncertainty
is bootstrapped over the six cancer-level paired effects.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "03_鏁版嵁涓庣粨鏋? / "processed_data"
LOCK = ROOT / "09_JBI_SUBMISSION_LOCK" / "final_config_comparison_ALL_LOCKED.json"
TABLES = ROOT / "07_璁烘枃鍒濈" / "tables"
OUT = ROOT / "09_JBI_SUBMISSION_LOCK" / "confidence_intervals"

# Public-package paths can be overridden without editing source.
DATA = Path(os.environ.get("MKG_DATA_ROOT", ROOT / "data" / "processed_data"))
LOCK = Path(
    os.environ.get(
        "MKG_LOCK_JSON",
        ROOT / "results" / "submission_lock" / "final_config_comparison_ALL_LOCKED.json",
    )
)
TABLES = Path(os.environ.get("MKG_TABLE_OUTPUT", ROOT / "results" / "source_tables"))
OUT = Path(os.environ.get("MKG_CI_OUTPUT", ROOT / "results" / "confidence_intervals"))

PRIMARY = {
    "LUAD": "GSE31210",
    "LIHC": "GSE14520",
    "KIRC": "GSE29609",
    "COAD": "GSE39582",
    "STAD": "GSE84437",
    "HNSC": "GSE65858",
}
METHODS = {
    "MKG": "GR-SAFS_v2",
    "Uni-Cox": "Uni-Cox",
    "Cox-Lasso": "Cox-Lasso",
    "Cox-EN": "Cox-EN",
    "RSF": "RSF",
    "DeepSurv": "DeepSurv",
}
PRIMARY_FOR_MEAN = ["LUAD", "LIHC", "COAD", "STAD", "HNSC"]
SEED = 20260726
N_PATIENT_BOOT = 1000
N_CANCER_BOOT = 50000


def frozen_risk(
    expr: pd.DataFrame, genes: list[str], directions: list[float], scores: list[float]
) -> tuple[np.ndarray, int]:
    available = [gene for gene in genes if gene in expr.columns]
    if not available:
        raise ValueError("No signature genes were available in the external cohort")
    indices = [genes.index(gene) for gene in available]
    matrix = expr[available].to_numpy(dtype=float)
    mean = np.nanmean(matrix, axis=0)
    sd = np.nanstd(matrix, axis=0)
    sd[sd < 1e-8] = 1.0
    standardized = np.nan_to_num((matrix - mean) / sd, nan=0.0)
    coefficient = np.asarray(directions, dtype=float)[indices] * np.asarray(scores, dtype=float)[indices]
    return standardized @ coefficient, len(available)


def load_external(cancer: str, cohort: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    directory = DATA / cancer / "external" / cohort
    expression = pd.read_csv(directory / "expr.tsv", sep="\t", index_col=0)
    expression.columns = expression.columns.astype(str)
    survival = pd.read_csv(directory / "survival.tsv", sep="\t")
    survival.columns = [column.lower().replace(".", "_") for column in survival.columns]
    sample_column = survival.columns[0]
    event_column = next(
        column for column in survival.columns if column in {"os", "event", "status", "os_event"}
    )
    time_column = next(
        column for column in survival.columns if column in {"os_time", "time", "survival_time"}
    )
    survival = survival.set_index(sample_column)
    common = sorted(set(expression.index) & set(survival.index))
    expression = expression.loc[common]
    event = pd.to_numeric(survival.loc[common, event_column], errors="coerce").to_numpy()
    time = pd.to_numeric(survival.loc[common, time_column], errors="coerce").to_numpy()
    valid = np.isfinite(event) & np.isfinite(time) & (time > 0)
    return expression.loc[valid], time[valid].astype(float), event[valid].astype(int)


def c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    return float(
        concordance_index_censored(
            np.asarray(event, dtype=bool),
            np.asarray(time, dtype=float),
            np.asarray(risk, dtype=float),
        )[0]
    )


def patient_bootstrap_ci(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_PATIENT_BOOT,
) -> tuple[float, float, int]:
    n = len(time)
    values: list[float] = []
    for _ in range(n_boot):
        index = rng.integers(0, n, size=n)
        try:
            value = c_index(time[index], event[index], risk[index])
        except ZeroDivisionError:
            continue
        if np.isfinite(value):
            values.append(value)
    if len(values) < int(0.9 * n_boot):
        raise RuntimeError(f"Only {len(values)} valid bootstrap samples out of {n_boot}")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high), len(values)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(N_CANCER_BOOT, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def external_ci(lock_data: dict) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    risks_by_cancer: dict[str, dict[str, np.ndarray]] = {}
    outcomes: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for cancer, cohort in PRIMARY.items():
        expression, time, event = load_external(cancer, cohort)
        outcomes[cancer] = (time, event)
        risks_by_cancer[cancer] = {}
        for display_name, lock_name in METHODS.items():
            signature = lock_data[cancer]["method_signatures"][lock_name]
            risk, matched = frozen_risk(
                expression,
                signature["genes"],
                signature["directions"],
                signature["scores"],
            )
            risks_by_cancer[cancer][display_name] = risk
            point = c_index(time, event, risk)
            rng = np.random.default_rng(SEED + sum(map(ord, cancer + display_name)))
            low, high, valid_boot = patient_bootstrap_ci(time, event, risk, rng)
            locked_point = float(lock_data[cancer]["external"][cohort][lock_name]["c_index"])
            if abs(point - locked_point) > 1e-4:
                raise AssertionError(
                    f"{cancer}/{cohort}/{display_name}: reconstructed {point} != locked {locked_point}"
                )
            rows.append(
                {
                    "Cancer": cancer,
                    "External cohort": cohort,
                    "Role": "Stress test" if cancer == "KIRC" else "Primary",
                    "Method": display_name,
                    "n": len(time),
                    "Events": int(event.sum()),
                    "Matched Top-20 genes": matched,
                    "C-index": point,
                    "CI low": low,
                    "CI high": high,
                    "Valid bootstrap replicates": valid_boot,
                }
            )

    frame = pd.DataFrame(rows)
    mean_summary: dict[str, dict] = {}
    rng = np.random.default_rng(SEED)
    for method in METHODS:
        values = frame[
            frame["Cancer"].isin(PRIMARY_FOR_MEAN) & (frame["Method"] == method)
        ]["C-index"].to_numpy()
        low, high = bootstrap_mean_ci(values, rng)
        mean_summary[method] = {
            "mean_external_c_index": float(values.mean()),
            "cancer_bootstrap_95_ci": [low, high],
            "n_primary_cohorts": len(values),
        }

    paired = []
    for cancer in PRIMARY_FOR_MEAN:
        mkgrisk = risks_by_cancer[cancer]["MKG"]
        unirisk = risks_by_cancer[cancer]["Uni-Cox"]
        time, event = outcomes[cancer]
        rng_cancer = np.random.default_rng(SEED + sum(map(ord, cancer)))
        differences = []
        for _ in range(N_PATIENT_BOOT):
            index = rng_cancer.integers(0, len(time), size=len(time))
            try:
                differences.append(
                    c_index(time[index], event[index], mkgrisk[index])
                    - c_index(time[index], event[index], unirisk[index])
                )
            except ZeroDivisionError:
                continue
        low, high = np.quantile(differences, [0.025, 0.975])
        paired.append(
            {
                "cancer": cancer,
                "MKG_minus_Uni_Cox": c_index(time, event, mkgrisk)
                - c_index(time, event, unirisk),
                "patient_bootstrap_95_ci": [float(low), float(high)],
            }
        )
    mean_summary["paired_MKG_minus_Uni_Cox"] = paired
    return frame, mean_summary


def stability_ci() -> dict:
    table = pd.read_csv(TABLES / "Table1_stability.csv")
    table = table[table["Cancer"] != "Mean"].copy()
    mk_col = next(column for column in table.columns if column.startswith("MKG"))
    cox_col = next(column for column in table.columns if column.startswith("Cox-Lasso"))
    mk = table[mk_col].to_numpy(dtype=float)
    cox = table[cox_col].to_numpy(dtype=float)
    difference = mk - cox
    ratio = mk / cox
    rng = np.random.default_rng(SEED)
    diff_low, diff_high = bootstrap_mean_ci(difference, rng)
    ratio_low, ratio_high = bootstrap_mean_ci(ratio, rng)
    return {
        "analysis_unit": "cancer",
        "n_cancers": len(table),
        "mean_MKG_normalized_RBO20": float(mk.mean()),
        "mean_Cox_Lasso_normalized_RBO20": float(cox.mean()),
        "mean_paired_difference": float(difference.mean()),
        "mean_paired_difference_cancer_bootstrap_95_ci": [diff_low, diff_high],
        "mean_ratio": float(ratio.mean()),
        "mean_ratio_cancer_bootstrap_95_ci": [ratio_low, ratio_high],
        "per_cancer_difference": dict(zip(table["Cancer"], difference.tolist())),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    lock_data = json.loads(LOCK.read_text(encoding="utf-8"))
    external, external_summary = external_ci(lock_data)
    stability_summary = stability_ci()

    external.to_csv(OUT / "MKG_LOCKED_EXTERNAL_CINDEX_BOOTSTRAP_CI.csv", index=False)
    external.to_csv(TABLES / "TableS_locked_external_cindex_ci.csv", index=False)
    summary = {
        "seed": SEED,
        "patient_bootstrap_replicates": N_PATIENT_BOOT,
        "cancer_bootstrap_replicates": N_CANCER_BOOT,
        "primary_external_cohorts": PRIMARY,
        "external_summary": external_summary,
        "stability_summary": stability_summary,
    }
    (OUT / "MKG_SUBMISSION_CONFIDENCE_INTERVALS.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(external.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

