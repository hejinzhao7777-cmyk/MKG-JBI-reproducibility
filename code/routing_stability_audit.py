"""
Repeated routing-stability audit for the JBI submission lock.

This script repeats only the graph-layer routing step under different CV/random
seeds. It does not rerun external validation or all prediction baselines. The
goal is to quantify whether the learned layer allocation is stable enough to
support the central relation-routing claim.

Outputs:
  - MKG_ROUTING_STABILITY_AUDIT.csv: one row per cancer and repeat
  - MKG_ROUTING_STABILITY_SUMMARY.csv: dominant-layer/no-relation frequencies
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sksurv.util import Surv

import final_config_comparison as F


DEFAULT_CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
LAYERS = ["coexpr", "meth", "cnv"]


def configure_from_args(args):
    F.N_BOOTSTRAP = int(args.bootstrap)
    F.STAGE1_RF_TREES = int(args.stage1_rf_trees)
    F.RF_JOBS = int(args.rf_jobs)
    F.PGD_MAX_ITER = int(args.pgd_max_iter)
    F.PGD_POWER_ITER = int(args.pgd_power_iter)


def load_cancer(cancer):
    cdir = F.ROOT / cancer
    expr = pd.read_csv(cdir / "expr_final.tsv", sep="\t", index_col=0)
    expr.columns = [str(c) for c in expr.columns]
    dev = pd.read_csv(cdir / "deviance_residuals.tsv", sep="\t", index_col=0)
    clin_p = cdir / "clinical_covariates.tsv"
    clin = pd.read_csv(clin_p, sep="\t", index_col=0) if clin_p.exists() else pd.DataFrame(index=expr.index)

    common = sorted(set(expr.index) & set(dev.index) & set(clin.index))
    expr_c = expr.loc[common]
    if expr_c.isna().any().any():
        expr_c = expr_c.fillna(expr_c.mean()).fillna(0.0)

    G = expr_c.values.astype(np.float64)
    y = dev.loc[common, "deviance_residual"].values.astype(np.float64)
    yt = dev.loc[common, "OS_time"].values.astype(float)
    ye = dev.loc[common, "OS"].values.astype(int)
    ysurv = Surv.from_arrays(ye.astype(bool), yt)
    clin_c = clin.loc[common]
    p = G.shape[1]

    Ls = {}
    for layer, filename in [("coexpr", "L_coexpr.npz"), ("meth", "L_meth_expr.npz"), ("cnv", "L_cnv.npz")]:
        path = cdir / "graph" / filename
        if path.exists():
            Ls[layer] = F.normalize_laplacian(sparse.load_npz(str(path)), p)
    Lzero = sparse.csr_matrix((p, p), dtype=np.float32)
    return G, y, ysurv, clin_c, Ls, Lzero


def route_once(G, y, ysurv, clin_c, Ls, Lzero, seed):
    F.SEED = int(seed)
    np.random.seed(int(seed))

    s_base = F.bootstrap_stability(G, y, Lzero, seed=seed)
    stabilities = {k: F.bootstrap_stability(G, y, L, seed=seed) for k, L in Ls.items()}

    base_top, _, _, _, _ = F.stage1_select(G, y, Lzero)
    ci_base = F.stage2_oof_ci(G[:, base_top], clin_c, ysurv)

    deltas = {}
    stage2_cis = {}
    for k, L in Ls.items():
        top, _, _, _, _ = F.stage1_select(G, y, L)
        stage2_cis[k] = F.stage2_oof_ci(G[:, top], clin_c, ysurv)
        deltas[k] = stage2_cis[k] - ci_base

    raw = {k: stabilities[k] * max(deltas[k], 0.0) for k in Ls}
    total = sum(raw.values())
    if total < 1e-15:
        weights = {k: 0.0 for k in Ls}
        mode = "reject_all_graphs"
        dominant = "no_relation"
    else:
        weights = {k: raw[k] / total for k in Ls}
        mode = "dual_driven"
        dominant = max(weights, key=weights.get)

    return {
        "weight_mode": mode,
        "dominant_layer": dominant,
        "baseline_stability": float(s_base),
        "baseline_ci": float(ci_base),
        "weights": weights,
        "stabilities": stabilities,
        "deltas": deltas,
        "stage2_cis": stage2_cis,
    }


def summarize(df):
    rows = []
    for cancer, sub in df.groupby("cancer"):
        row = {
            "cancer": cancer,
            "n_repeats": int(len(sub)),
            "no_relation_frequency": float((sub["dominant_layer"] == "no_relation").mean()),
        }
        counts = sub["dominant_layer"].value_counts(normalize=True)
        row["dominant_layer_mode"] = str(counts.index[0])
        row["dominant_layer_mode_frequency"] = float(counts.iloc[0])
        for layer in LAYERS:
            row[f"{layer}_dominant_frequency"] = float((sub["dominant_layer"] == layer).mean())
            row[f"w_{layer}_mean"] = float(sub[f"w_{layer}"].mean())
            row[f"w_{layer}_sd"] = float(sub[f"w_{layer}"].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancers", default=",".join(DEFAULT_CANCERS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed0", type=int, default=4200)
    parser.add_argument("--bootstrap", type=int, default=int(os.environ.get("FINAL_BOOTSTRAP", "30")))
    parser.add_argument("--stage1-rf-trees", type=int, default=int(os.environ.get("FINAL_STAGE1_RF_TREES", "300")))
    parser.add_argument("--rf-jobs", type=int, default=int(os.environ.get("FINAL_RF_JOBS", "8")))
    parser.add_argument("--pgd-max-iter", type=int, default=int(os.environ.get("FINAL_PGD_MAX_ITER", "300")))
    parser.add_argument("--pgd-power-iter", type=int, default=int(os.environ.get("FINAL_PGD_POWER_ITER", "8")))
    parser.add_argument("--outdir", default=str(F.OUT.parent / "09_JBI_SUBMISSION_LOCK"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    configure_from_args(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "MKG_ROUTING_STABILITY_AUDIT.csv"
    summary_path = outdir / "MKG_ROUTING_STABILITY_SUMMARY.csv"
    meta_path = outdir / "MKG_ROUTING_STABILITY_AUDIT_meta.json"

    rows = []
    if args.resume and raw_path.exists():
        rows = pd.read_csv(raw_path).to_dict("records")

    done = {(r["cancer"], int(r["repeat"])) for r in rows}
    cancers = [c.strip() for c in args.cancers.split(",") if c.strip()]
    seeds = [args.seed0 + i for i in range(args.repeats)]

    for cancer in cancers:
        print(f"\n[{cancer}] loading data", flush=True)
        G, y, ysurv, clin_c, Ls, Lzero = load_cancer(cancer)
        for repeat, seed in enumerate(seeds, start=1):
            if (cancer, repeat) in done:
                print(f"[{cancer}] repeat {repeat}/{args.repeats}: skip existing", flush=True)
                continue
            print(f"[{cancer}] repeat {repeat}/{args.repeats}, seed={seed}", flush=True)
            res = route_once(G, y, ysurv, clin_c, Ls, Lzero, seed)
            row = {
                "cancer": cancer,
                "repeat": repeat,
                "seed": seed,
                "weight_mode": res["weight_mode"],
                "dominant_layer": res["dominant_layer"],
                "baseline_stability": res["baseline_stability"],
                "baseline_ci": res["baseline_ci"],
            }
            for layer in LAYERS:
                row[f"w_{layer}"] = float(res["weights"].get(layer, 0.0))
                row[f"stability_{layer}"] = float(res["stabilities"].get(layer, np.nan))
                row[f"delta_{layer}"] = float(res["deltas"].get(layer, np.nan))
                row[f"stage2_ci_{layer}"] = float(res["stage2_cis"].get(layer, np.nan))
            rows.append(row)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            summarize(pd.DataFrame(rows)).to_csv(summary_path, index=False)

    df = pd.DataFrame(rows)
    summary = summarize(df)
    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    meta = {
        "cancers": cancers,
        "repeats": args.repeats,
        "seeds": seeds,
        "bootstrap_B": F.N_BOOTSTRAP,
        "stage1_rf_trees": F.STAGE1_RF_TREES,
        "rf_jobs": F.RF_JOBS,
        "pgd_max_iter": F.PGD_MAX_ITER,
        "pgd_power_iter": F.PGD_POWER_ITER,
        "outputs": {"raw": str(raw_path), "summary": str(summary_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nsaved {raw_path}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()

