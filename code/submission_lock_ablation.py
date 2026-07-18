"""
Submission-lock ablation consolidation for MKG.

The original ablation script evaluated external transfer after fitting a new
ridge-Cox model on each selected gene set. The submission manuscript now uses a
frozen direction-score molecular risk score for external C-index. This script
recomputes the ablation table under that exact locked scoring protocol.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import final_config_comparison as F


CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
PRIMARY_EXTERNAL = {
    "LUAD": "GSE31210",
    "LIHC": "GSE14520",
    "KIRC": "GSE29609",
    "COAD": "GSE39582",
    "STAD": "GSE84437",
    "HNSC": "GSE65858",
}
METHOD_MAP = {
    "single_coexpr": "GR-SAFS_v1",
    "equal": "v2_equal",
    "joint": "GR-SAFS_v2",
}
OUTDIR = F.OUT.parent / "09_JBI_SUBMISSION_LOCK"


def load_cancer_expression(cancer):
    cdir = F.ROOT / cancer
    expr = pd.read_csv(cdir / "expr_final.tsv", sep="\t", index_col=0)
    expr.columns = [str(c) for c in expr.columns]
    dev = pd.read_csv(cdir / "deviance_residuals.tsv", sep="\t", index_col=0)
    common = sorted(set(expr.index) & set(dev.index))
    expr_c = expr.loc[common]
    if expr_c.isna().any().any():
        expr_c = expr_c.fillna(expr_c.mean()).fillna(0.0)
    y = dev.loc[common, "deviance_residual"].values.astype(np.float64)
    names = np.array([str(g) for g in expr.columns])
    return cdir, expr_c, y, names


def load_external(cancer):
    ed = F.ROOT / cancer / "external" / PRIMARY_EXTERNAL[cancer]
    expr = pd.read_csv(ed / "expr.tsv", sep="\t", index_col=0)
    expr.columns = [str(c) for c in expr.columns]
    surv = pd.read_csv(ed / "survival.tsv", sep="\t")
    surv.columns = [c.lower().replace(".", "_") for c in surv.columns]
    event_col = next(c for c in surv.columns if c in ["os", "event", "status", "os_event"])
    time_col = next(c for c in surv.columns if c in ["os_time", "time", "survival_time"])
    surv = surv.set_index(surv.columns[0])
    common = sorted(set(expr.index) & set(surv.index))
    expr = expr.loc[common]
    event = pd.to_numeric(surv.loc[common, event_col], errors="coerce").values
    time = pd.to_numeric(surv.loc[common, time_col], errors="coerce").values
    ok = (~np.isnan(event)) & (~np.isnan(time)) & (time > 0)
    return expr.loc[ok], time[ok], event[ok].astype(int)


def external_cindex_from_signature(cancer, sig):
    expr, time, event = load_external(cancer)
    risk, nmatch = F.frozen_risk(expr, sig["genes"], sig["directions"], sig["scores"])
    if risk is None:
        return None, 0
    return F.evaluate(risk, time, event)["c_index"], nmatch


def stability_only_signature(cancer):
    cdir, expr_c, y, names = load_cancer_expression(cancer)
    p = expr_c.shape[1]
    locked = json.load(open(F.OUT / f"final_config_comparison_{cancer}.json", encoding="utf-8"))[cancer]
    stabilities = locked["stabilities"]
    total = sum(stabilities.values())
    weights = {k: stabilities[k] / total for k in stabilities}
    layer_files = {"coexpr": "L_coexpr.npz", "meth": "L_meth_expr.npz", "cnv": "L_cnv.npz"}
    Ls = {
        k: F.normalize_laplacian(sparse.load_npz(str(cdir / "graph" / layer_files[k])), p)
        for k in weights
    }
    Lw = sum(weights[k] * Ls[k] for k in weights)
    top, dirs_full, scores, _, nnz = F.stage1_select(expr_c.values.astype(np.float64), y, Lw)
    return {
        "genes": names[top].tolist(),
        "directions": dirs_full[top].tolist(),
        "scores": scores[top].tolist(),
        "nnz": int(nnz),
    }


def main():
    rows = []
    result = {}
    for cancer in CANCERS:
        locked = json.load(open(F.OUT / f"final_config_comparison_{cancer}.json", encoding="utf-8"))[cancer]
        result[cancer] = {}
        for scheme, method in METHOD_MAP.items():
            ci = locked["external"][PRIMARY_EXTERNAL[cancer]][method]["c_index"]
            result[cancer][scheme] = float(ci)
            rows.append({
                "Cancer": cancer,
                "External cohort": PRIMARY_EXTERNAL[cancer],
                "Scheme": scheme,
                "c_index": float(ci),
                "source": f"locked method {method}",
            })
        sig = stability_only_signature(cancer)
        ci, nmatch = external_cindex_from_signature(cancer, sig)
        result[cancer]["stability_only"] = float(ci) if ci is not None else None
        rows.append({
            "Cancer": cancer,
            "External cohort": PRIMARY_EXTERNAL[cancer],
            "Scheme": "stability_only",
            "c_index": float(ci) if ci is not None else np.nan,
            "n_matched": nmatch,
            "source": "recomputed stability-only signature with frozen direction-score risk",
        })
        print(
            f"{cancer}: "
            + ", ".join(f"{k}={result[cancer][k]:.4f}" for k in ["single_coexpr", "equal", "stability_only", "joint"])
        )

    mean_rows = []
    for scheme in ["single_coexpr", "equal", "stability_only", "joint"]:
        vals = [result[c][scheme] for c in CANCERS if c != "KIRC" and result[c][scheme] is not None]
        result.setdefault("_mean_exclKIRC", {})[scheme] = float(np.mean(vals))
        mean_rows.append({
            "Cancer": "Mean excl. KIRC",
            "External cohort": "-",
            "Scheme": scheme,
            "c_index": float(np.mean(vals)),
            "source": "mean over LUAD, LIHC, COAD, STAD, HNSC",
        })

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "MKG_SUBMISSION_LOCK_ABLATION.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    df = pd.DataFrame(rows + mean_rows)
    df.to_csv(OUTDIR / "MKG_SUBMISSION_LOCK_ABLATION.csv", index=False)
    df.to_csv(F.OUT / "submission_lock_ablation.csv", index=False)
    (F.OUT / "submission_lock_ablation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved {OUTDIR / 'MKG_SUBMISSION_LOCK_ABLATION.csv'}")


if __name__ == "__main__":
    main()

