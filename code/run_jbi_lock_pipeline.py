"""
JBI lock pipeline for MKG.

This script creates a single reproducible result version for the JBI manuscript:
  1. run final_config_comparison.py for the locked six-cancer panel;
  2. consolidate weights, Top-20 signatures, training C-index, and external C-index;
  3. regenerate JBI-specific audit figures;
  4. optionally run the fast train-only graph nested audit;
  5. compile the manuscript and write a manifest with hashes and timestamps.

Usage:
  python run_jbi_lock_pipeline.py
  python run_jbi_lock_pipeline.py --skip-final-config
  python run_jbi_lock_pipeline.py --final-bootstrap 10
  python run_jbi_lock_pipeline.py --nested-cancers LUAD,COAD,LIHC
  python run_jbi_lock_pipeline.py --lock-name MKG_FAST_AUDIT_LOCK --lock-dir 08_FAST_AUDIT_LOCK
  python run_jbi_lock_pipeline.py --lock-name MKG_JBI_SUBMISSION_LOCK --lock-dir 09_JBI_SUBMISSION_LOCK
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANUSCRIPT = ROOT / "07_论文初稿"
TABLES = MANUSCRIPT / "tables"
FIGURES = MANUSCRIPT / "figures"
LOCK_DIR = ROOT / "08_JBI_lock"
LOCK_TABLES = LOCK_DIR / "tables"
LOCK_LOGS = LOCK_DIR / "logs"

CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
PRIMARY_EXTERNAL = {
    "LUAD": "GSE31210",
    "LIHC": "GSE14520",
    "KIRC": "GSE29609",
    "COAD": "GSE39582",
    "STAD": "GSE84437",
    "HNSC": "GSE65858",
}
METHOD_LABELS = {
    "GR-SAFS_v2": "MKG",
    "GR-SAFS_v1": "GR-SAFS_coexpr",
    "v2_equal": "MKG_equal",
}


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_lock(lock_dir: Path) -> None:
    global LOCK_DIR, LOCK_TABLES, LOCK_LOGS
    LOCK_DIR = lock_dir
    LOCK_TABLES = LOCK_DIR / "tables"
    LOCK_LOGS = LOCK_DIR / "logs"


def run_step(cmd: list[str], log_name: str, cwd: Path = HERE, env: dict[str, str] | None = None) -> int:
    LOCK_LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOCK_LOGS / log_name
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\nstarted: {now_iso()}\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.write(f"\nfinished: {now_iso()}\nexit_code: {proc.returncode}\n")
    print(f"[{proc.returncode}] {' '.join(cmd)} -> {log_path}", flush=True)
    return proc.returncode


def load_final_results() -> dict:
    results = {}
    missing = []
    for cancer in CANCERS:
        path = HERE / f"final_config_comparison_{cancer}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        data = json.load(path.open("r", encoding="utf-8"))
        if cancer not in data or "error" in data.get(cancer, {}):
            missing.append(str(path))
            continue
        results[cancer] = data[cancer]
    if missing:
        raise RuntimeError("Missing or invalid locked final-config files:\n" + "\n".join(missing))
    return results


def write_csv(df: pd.DataFrame, name: str) -> Path:
    LOCK_TABLES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    out = LOCK_TABLES / name
    df.to_csv(out, index=False, encoding="utf-8-sig")
    df.to_csv(TABLES / name, index=False, encoding="utf-8-sig")
    return out


def consolidate_final_config(results: dict) -> dict[str, str]:
    outputs = {}
    combined_path = LOCK_DIR / "final_config_comparison_ALL_LOCKED.json"
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(results, combined_path.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)
    outputs["combined_final_config"] = str(combined_path)

    weight_rows = []
    train_rows = []
    external_rows = []
    top_rows = []
    for cancer, obj in results.items():
        weights = obj.get("omics_weights", {})
        stabilities = obj.get("stabilities", {})
        deltas = obj.get("deltas", {})
        row = {
            "Cancer": cancer,
            "weight_mode": obj.get("weight_mode", ""),
            "baseline_ci_no_relation": obj.get("baseline_ci", ""),
        }
        for layer in ["coexpr", "meth", "cnv"]:
            row[f"w_{layer}"] = weights.get(layer, 0.0)
            row[f"stability_{layer}"] = stabilities.get(layer, "")
            row[f"delta_{layer}"] = deltas.get(layer, "")
        weight_rows.append(row)

        for method, metrics in obj.get("training", {}).items():
            train_rows.append({
                "Cancer": cancer,
                "Method": METHOD_LABELS.get(method, method),
                "c_index": metrics.get("c_index", ""),
                "km_p": metrics.get("km_p", ""),
                "auc_1y": metrics.get("auc_1y", ""),
            })

        ext_name = PRIMARY_EXTERNAL[cancer]
        for method, metrics in obj.get("external", {}).get(ext_name, {}).items():
            external_rows.append({
                "Cancer": cancer,
                "External cohort": ext_name,
                "Method": METHOD_LABELS.get(method, method),
                "c_index": metrics.get("c_index", ""),
                "km_p": metrics.get("km_p", ""),
                "auc_1y": metrics.get("auc_1y", ""),
                "n_matched": metrics.get("n_matched", ""),
            })

        for method, genes in obj.get("method_genes", {}).items():
            for rank, gene in enumerate(genes, start=1):
                top_rows.append({
                    "Cancer": cancer,
                    "Method": METHOD_LABELS.get(method, method),
                    "Rank": rank,
                    "Gene": gene,
                })

    outputs["weights"] = str(write_csv(pd.DataFrame(weight_rows), "Table_JBI_LOCK_weights.csv"))
    outputs["training_cindex"] = str(write_csv(pd.DataFrame(train_rows), "Table_JBI_LOCK_training_cindex.csv"))
    outputs["external_cindex"] = str(write_csv(pd.DataFrame(external_rows), "Table_JBI_LOCK_external_cindex.csv"))
    outputs["top20"] = str(write_csv(pd.DataFrame(top_rows), "Table_JBI_LOCK_top20.csv"))
    nested_table = LOCK_TABLES / "Table_JBI_LOCK_nested_train_only_audit.csv"
    if nested_table.exists():
        outputs["nested_train_only_audit"] = str(nested_table)
    return outputs


def write_manifest(outputs: dict, step_codes: dict, args: argparse.Namespace) -> Path:
    tracked = [
        HERE / "final_config_comparison.py",
        HERE / "nested_deleakage.py",
        HERE / "routing_reliability_simulation.py",
        HERE / "run_jbi_lock_pipeline.py",
        HERE / "redraw_fig2_submission_weights.py",
        HERE / "redraw_fig3_submission_generalization.py",
        HERE / "submission_delta_audit.py",
        MANUSCRIPT / "MKG_JBI.tex",
        MANUSCRIPT / "MKG_JBI.pdf",
        MANUSCRIPT / "MKG_JBI_LOCK.pdf",
        MANUSCRIPT / "MKG_JBI_Supplementary_Methods.tex",
        MANUSCRIPT / "MKG_JBI_Supplementary_Methods.pdf",
        MANUSCRIPT / "MKG_JBI_elsarticle.tex",
        MANUSCRIPT / "MKG_JBI_elsarticle.pdf",
        MANUSCRIPT / "make_elsarticle_jbi.py",
        LOCK_DIR / "JBI_reproducibility_checklist.md",
        LOCK_TABLES / "Table_JBI_LOCK_nested_train_only_audit.csv",
        FIGURES / "Fig1_MKG_complete_framework.pdf",
        FIGURES / "Fig2_omics_weights.pdf",
        FIGURES / "Fig3_generalization.pdf",
        FIGURES / "FigJBI_routing_reliability_simulation.pdf",
        FIGURES / "FigJBI_expanded_stability_baselines.pdf",
        FIGURES / "Fig4_stability.pdf",
    ]
    for cancer in CANCERS:
        tracked.append(HERE / f"final_config_comparison_{cancer}.json")

    manifest = {
        "lock_name": args.lock_name,
        "created_at": now_iso(),
        "cancers": CANCERS,
        "primary_external": PRIMARY_EXTERNAL,
        "final_config": {
            "lambda1": 0.2,
            "lambda2": 50.0,
            "gamma": 10.0,
            "top_k": 20,
            "bootstrap_B": int(args.final_bootstrap),
            "stage1_rf_trees": int(args.final_stage1_rf_trees),
            "rsf_trees": int(args.final_rsf_trees),
            "deepsurv_epochs": int(args.final_deepsurv_epochs),
            "pgd_max_iter": int(args.final_pgd_max_iter),
            "pgd_power_iter": int(args.final_pgd_power_iter),
            "rbo": "normalized truncated RBO@20, p=0.9",
            "no_relation_baseline": "zero Laplacian L0=0",
            "negative_gain_rule": "reject all graph layers and use no-relation selector",
        },
        "step_exit_codes": step_codes,
        "outputs": outputs,
        "hashes": {str(p): sha256(p) for p in tracked if p.exists()},
    }
    out = LOCK_DIR / f"{args.lock_name}_manifest.json"
    json.dump(manifest, out.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-name", default="MKG_JBI_LOCK")
    ap.add_argument("--lock-dir", default="08_JBI_lock")
    ap.add_argument("--skip-final-config", action="store_true")
    ap.add_argument("--skip-nested", action="store_true")
    ap.add_argument("--final-bootstrap", default="30")
    ap.add_argument("--final-rf-jobs", default="8")
    ap.add_argument("--final-stage1-rf-trees", default="300")
    ap.add_argument("--final-rsf-trees", default="100")
    ap.add_argument("--final-deepsurv-epochs", default="50")
    ap.add_argument("--final-pgd-max-iter", default="300")
    ap.add_argument("--final-pgd-power-iter", default="8")
    ap.add_argument("--nested-cancers", default="LUAD,COAD,LIHC")
    ap.add_argument("--nested-bootstrap", default="10")
    ap.add_argument("--compile", action="store_true", default=True)
    args = ap.parse_args()

    configure_lock(ROOT / args.lock_dir)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    step_codes = {}

    if not args.skip_final_config:
        env_final = os.environ.copy()
        env_final["FINAL_BOOTSTRAP"] = str(args.final_bootstrap)
        env_final["FINAL_RF_JOBS"] = str(args.final_rf_jobs)
        env_final["FINAL_STAGE1_RF_TREES"] = str(args.final_stage1_rf_trees)
        env_final["FINAL_RSF_TREES"] = str(args.final_rsf_trees)
        env_final["FINAL_DEEPSURV_EPOCHS"] = str(args.final_deepsurv_epochs)
        env_final["FINAL_PGD_MAX_ITER"] = str(args.final_pgd_max_iter)
        env_final["FINAL_PGD_POWER_ITER"] = str(args.final_pgd_power_iter)
        for cancer in CANCERS:
            code = run_step(
                [sys.executable, "final_config_comparison.py", cancer],
                f"final_config_{cancer}.log",
                env=env_final,
            )
            step_codes[f"final_config_{cancer}"] = code
            if code != 0:
                raise SystemExit(code)

    results = load_final_results()
    outputs = consolidate_final_config(results)

    env_lock = os.environ.copy()
    env_lock["MKG_LOCK_DIR"] = str(LOCK_DIR)
    for script in [
        "redraw_fig2_submission_weights.py",
        "redraw_fig3_submission_generalization.py",
        "routing_reliability_simulation.py",
        "expanded_stability_baselines_jbi.py",
        "redraw_fig4_stability_normalized.py",
        "generate_fig1_jbi_framework.py",
    ]:
        code = run_step([sys.executable, script], f"{Path(script).stem}.log", env=env_lock)
        step_codes[Path(script).stem] = code
        if code != 0:
            raise SystemExit(code)

    if not args.skip_nested:
        env = os.environ.copy()
        env["NESTED_CANCERS"] = args.nested_cancers
        env["NESTED_BOOTSTRAP"] = args.nested_bootstrap
        env["NESTED_RESUME"] = "0"
        code = run_step([sys.executable, "nested_deleakage.py"], "nested_deleakage.log", env=env)
        step_codes["nested_deleakage"] = code
        if code != 0:
            raise SystemExit(code)

    pdflatex = shutil.which("pdflatex")
    if args.compile and pdflatex:
        for i in range(3):
            code = run_step(
                [pdflatex, "-interaction=nonstopmode", "-jobname=MKG_JBI_LOCK", "MKG_JBI.tex"],
                f"pdflatex_{i+1}.log",
                cwd=MANUSCRIPT,
            )
            step_codes[f"pdflatex_{i+1}"] = code
            if code != 0:
                raise SystemExit(code)
    elif args.compile:
        step_codes["pdflatex"] = "not_found"

    manifest = write_manifest(outputs, step_codes, args)
    print(f"LOCK MANIFEST: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

