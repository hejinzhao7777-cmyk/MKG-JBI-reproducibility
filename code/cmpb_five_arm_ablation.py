"""
CMPB five-arm routing ablation for the locked MKG analysis.

The experiment isolates the two routing signals under one protocol:
  1. no_graph: zero Laplacian;
  2. equal: equal weights over all available graph layers;
  3. stability_only: weights proportional to bootstrap stability;
  4. utility_only: weights proportional to positive OOF C-index gain;
  5. joint: weights proportional to stability * positive OOF gain.

All signatures are selected from the TCGA cohort. External cohorts are used
only after the genes, directions, and scores have been frozen. Results are
saved incrementally so a long run can be resumed without overwriting the
previous JBI submission lock.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sksurv.util import Surv

import final_config_comparison as F


CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
PRIMARY_CANCERS = ["LUAD", "LIHC", "COAD", "STAD", "HNSC"]
PRIMARY_EXTERNAL = {
    "LUAD": "GSE31210",
    "LIHC": "GSE14520",
    "KIRC": "GSE29609",
    "COAD": "GSE39582",
    "STAD": "GSE84437",
    "HNSC": "GSE65858",
}
SCHEMES = ["no_graph", "equal", "stability_only", "utility_only", "joint"]
LAYER_FILES = {
    "coexpr": "L_coexpr.npz",
    "meth": "L_meth_expr.npz",
    "cnv": "L_cnv.npz",
}
OUTDIR = F.OUT.parent / "10_CMPB_SUBMISSION_LOCK"
TARGET_CANCERS = sys.argv[1:] if len(sys.argv) > 1 else CANCERS
RUN_SUFFIX = "" if TARGET_CANCERS == CANCERS else "_" + "_".join(TARGET_CANCERS)
CACHE_PATH = OUTDIR / f"CMPB_FIVE_ARM_ABLATION_CACHE{RUN_SUFFIX}.json"
TABLE_PATH = OUTDIR / f"CMPB_FIVE_ARM_ABLATION{RUN_SUFFIX}.csv"
STABILITY_RAW_PATH = OUTDIR / f"CMPB_FIVE_ARM_STABILITY_RAW{RUN_SUFFIX}.json"
STABILITY_TABLE_PATH = OUTDIR / f"CMPB_FIVE_ARM_STABILITY{RUN_SUFFIX}.csv"
RESOURCE_PATH = OUTDIR / f"CMPB_COMPUTATIONAL_COST{RUN_SUFFIX}.csv"
LEGACY_STABILITY_PATH = (
    F.OUT.parent
    / "09_JBI_SUBMISSION_LOCK"
    / "MKG_SUBMISSION_LOCK_ABLATION_STABILITY_RAW.json"
)
LEGACY_ABLATION_PATH = (
    F.OUT.parent / "09_JBI_SUBMISSION_LOCK" / "MKG_SUBMISSION_LOCK_ABLATION.csv"
)
COMPUTE_INTERNAL_OOF = os.environ.get("CMPB_INTERNAL_OOF", "0") == "1"


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_cancer(cancer: str):
    cdir = F.ROOT / cancer
    expr = pd.read_csv(cdir / "expr_final.tsv", sep="\t", index_col=0)
    expr.columns = [str(c) for c in expr.columns]
    dev = pd.read_csv(cdir / "deviance_residuals.tsv", sep="\t", index_col=0)
    clin_path = cdir / "clinical_covariates.tsv"
    clin = (
        pd.read_csv(clin_path, sep="\t", index_col=0)
        if clin_path.exists()
        else pd.DataFrame(index=expr.index)
    )
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
    names = np.array([str(g) for g in expr.columns])
    p = G.shape[1]
    laps = {
        layer: F.normalize_laplacian(
            sparse.load_npz(str(cdir / "graph" / filename)), p
        )
        for layer, filename in LAYER_FILES.items()
        if (cdir / "graph" / filename).exists()
    }
    return cdir, expr_c, G, y, yt, ye, ysurv, clin_c, names, laps


def load_external(cancer: str):
    cohort = PRIMARY_EXTERNAL[cancer]
    edir = F.ROOT / cancer / "external" / cohort
    expr = pd.read_csv(edir / "expr.tsv", sep="\t", index_col=0)
    expr.columns = [str(c) for c in expr.columns]
    surv = pd.read_csv(edir / "survival.tsv", sep="\t")
    surv.columns = [c.lower().replace(".", "_") for c in surv.columns]
    event_col = next(c for c in surv.columns if c in ["os", "event", "status", "os_event"])
    time_col = next(c for c in surv.columns if c in ["os_time", "time", "survival_time"])
    surv = surv.set_index(surv.columns[0])
    common = sorted(set(expr.index) & set(surv.index))
    expr = expr.loc[common]
    event = pd.to_numeric(surv.loc[common, event_col], errors="coerce").values
    duration = pd.to_numeric(surv.loc[common, time_col], errors="coerce").values
    valid = (~np.isnan(event)) & (~np.isnan(duration)) & (duration > 0)
    return cohort, expr.loc[valid], duration[valid], event[valid].astype(int)


def normalize_positive(values: dict[str, float]):
    positive = {k: max(float(v), 0.0) for k, v in values.items()}
    total = sum(positive.values())
    if total <= 1e-15:
        return {k: 0.0 for k in values}, 1.0
    return {k: positive[k] / total for k in values}, 0.0


def scheme_weights(scheme: str, locked: dict, layers: list[str]):
    if scheme == "no_graph":
        return {k: 0.0 for k in layers}, 1.0
    if scheme == "equal":
        return {k: 1.0 / len(layers) for k in layers}, 0.0
    if scheme == "stability_only":
        values = {k: float(locked["stabilities"][k]) for k in layers}
        total = sum(values.values())
        return {k: values[k] / total for k in layers}, 0.0
    if scheme == "utility_only":
        return normalize_positive({k: float(locked["deltas"][k]) for k in layers})
    if scheme == "joint":
        weights = {k: float(locked["omics_weights"].get(k, 0.0)) for k in layers}
        no_relation = 1.0 if sum(abs(v) for v in weights.values()) <= 1e-15 else 0.0
        return weights, no_relation
    raise ValueError(f"Unknown scheme: {scheme}")


def weighted_laplacian(laps, weights):
    p = next(iter(laps.values())).shape[0]
    if sum(abs(v) for v in weights.values()) <= 1e-15:
        return sparse.csr_matrix((p, p), dtype=np.float32)
    return sum(float(weights[k]) * laps[k] for k in laps)


class MemoryMonitor:
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def fit_signature(G, y, L, names):
    started = time.perf_counter()
    with MemoryMonitor() as monitor:
        signature = F.sig_grsafs(G, y, L, names)
    elapsed = time.perf_counter() - started
    return signature, elapsed, monitor.peak / (1024 ** 3)


def signature_equal(a: dict, b: dict) -> bool:
    return (
        a.get("genes") == b.get("genes")
        and np.allclose(a.get("directions", []), b.get("directions", []))
        and np.allclose(a.get("scores", []), b.get("scores", []))
    )


def jaccard(a, b):
    aa, bb = set(a), set(b)
    return len(aa & bb) / len(aa | bb) if (aa | bb) else 0.0


def pairwise_metric(lists, fn):
    vals = []
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            vals.append(fn(lists[i], lists[j]))
    return float(np.mean(vals)) if vals else np.nan


def bootstrap_lists(G, y, L, seed=F.SEED):
    n = G.shape[0]
    sample_size = int(n * F.BOOTSTRAP_RATIO)
    rng = np.random.RandomState(seed)
    lists = []
    for _ in range(F.N_BOOTSTRAP):
        idx = rng.choice(n, sample_size, replace=False)
        beta = F.graph_lasso_pgd(
            G[idx], y[idx], L, max_iter=F.PGD_MAX_ITER, tol=1e-4
        )
        lists.append(
            np.argsort(np.abs(beta))[::-1][: F.TOP_K].astype(int).tolist()
        )
    return lists


def existing_locked_signature(locked: dict, scheme: str):
    key = {"equal": "v2_equal", "joint": "GR-SAFS_v2"}.get(scheme)
    if key is None:
        return None
    return locked["method_signatures"].get(key)


def write_tables(cache: dict, stability_raw: dict) -> None:
    rows = []
    resources = []
    stability_rows = []
    denom = 1.0 - F.RBO_P ** F.TOP_K
    for cancer in CANCERS:
        for scheme in SCHEMES:
            obj = cache.get(cancer, {}).get(scheme)
            if not obj:
                continue
            rows.append(
                {
                    "Cancer": cancer,
                    "External cohort": obj["external_cohort"],
                    "Scheme": scheme,
                    "w_no_relation": obj["no_relation_weight"],
                    "w_coexpr": obj["weights"].get("coexpr", 0.0),
                    "w_meth": obj["weights"].get("meth", 0.0),
                    "w_cnv": obj["weights"].get("cnv", 0.0),
                    "Internal OOF C-index": obj["internal_oof_cindex"],
                    "Training frozen-score C-index": obj["training_cindex"],
                    "External frozen-score C-index": obj["external_cindex"],
                    "Matched genes": obj["matched_genes"],
                    "Signature source": obj["signature_source"],
                }
            )
            if obj.get("fit_seconds") is not None:
                resources.append(
                    {
                        "Cancer": cancer,
                        "Scheme": scheme,
                        "n": obj["n"],
                        "p": obj["p"],
                        "signature_fit_seconds": obj["fit_seconds"],
                        "peak_process_RSS_GB": obj["peak_process_rss_gb"],
                        "CPU": platform.processor(),
                        "Python": platform.python_version(),
                    }
                )
            lists = stability_raw.get(cancer, {}).get(scheme, {}).get("top_lists")
            if lists:
                rbo = pairwise_metric(lists, F.rbo_score)
                stability_rows.append(
                    {
                        "Cancer": cancer,
                        "Scheme": scheme,
                        "RBO@20": rbo,
                        "normalized_RBO@20": rbo / denom if denom else rbo,
                        "Jaccard": pairwise_metric(lists, jaccard),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        mean_rows = []
        for scheme in SCHEMES:
            sub = df[
                (df["Cancer"].isin(PRIMARY_CANCERS)) & (df["Scheme"] == scheme)
            ]
            if sub.empty:
                continue
            mean_rows.append(
                {
                    "Cancer": "Mean excl. KIRC",
                    "External cohort": "-",
                    "Scheme": scheme,
                    "w_no_relation": np.nan,
                    "w_coexpr": np.nan,
                    "w_meth": np.nan,
                    "w_cnv": np.nan,
                    "Internal OOF C-index": sub["Internal OOF C-index"].mean(),
                    "Training frozen-score C-index": sub[
                        "Training frozen-score C-index"
                    ].mean(),
                    "External frozen-score C-index": sub[
                        "External frozen-score C-index"
                    ].mean(),
                    "Matched genes": np.nan,
                    "Signature source": "mean over five primary external cohorts",
                }
            )
        pd.concat([df, pd.DataFrame(mean_rows)], ignore_index=True).to_csv(
            TABLE_PATH, index=False
        )
    if resources:
        pd.DataFrame(resources).to_csv(RESOURCE_PATH, index=False)
    sdf = pd.DataFrame(stability_rows)
    if not sdf.empty:
        mean_rows = []
        for scheme in SCHEMES:
            sub = sdf[
                (sdf["Cancer"].isin(PRIMARY_CANCERS)) & (sdf["Scheme"] == scheme)
            ]
            if sub.empty:
                continue
            mean_rows.append(
                {
                    "Cancer": "Mean excl. KIRC",
                    "Scheme": scheme,
                    "RBO@20": sub["RBO@20"].mean(),
                    "normalized_RBO@20": sub["normalized_RBO@20"].mean(),
                    "Jaccard": sub["Jaccard"].mean(),
                }
            )
        pd.concat([sdf, pd.DataFrame(mean_rows)], ignore_index=True).to_csv(
            STABILITY_TABLE_PATH, index=False
        )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache = load_json(CACHE_PATH, {})
    stability_raw = load_json(STABILITY_RAW_PATH, {})
    legacy_raw = load_json(LEGACY_STABILITY_PATH, {})
    legacy_ablation = pd.read_csv(LEGACY_ABLATION_PATH)

    for cancer in TARGET_CANCERS:
        print(f"\n===== {cancer} =====", flush=True)
        (
            _,
            expr_c,
            G,
            y,
            yt,
            ye,
            ysurv,
            clin_c,
            names,
            laps,
        ) = load_cancer(cancer)
        locked = load_json(F.OUT / f"final_config_comparison_{cancer}.json", {})[cancer]
        external_cohort, external_expr, external_time, external_event = load_external(cancer)
        cache.setdefault(cancer, {})
        stability_raw.setdefault(cancer, {})

        for scheme in SCHEMES:
            weights, no_relation = scheme_weights(scheme, locked, list(laps))
            L = weighted_laplacian(laps, weights)
            print(
                f"  {scheme}: no-relation={no_relation:.0f}, "
                + ", ".join(f"{k}={weights[k]:.4f}" for k in weights),
                flush=True,
            )

            if scheme not in cache[cancer]:
                locked_signature = existing_locked_signature(locked, scheme)
                if scheme == "stability_only":
                    row = legacy_ablation[
                        (legacy_ablation["Cancer"] == cancer)
                        & (legacy_ablation["Scheme"] == "stability_only")
                    ].iloc[0]
                    cache[cancer][scheme] = {
                        "weights": weights,
                        "no_relation_weight": no_relation,
                        "signature": None,
                        "signature_source": (
                            "existing submission-lock stability-only signature result"
                        ),
                        "n": int(G.shape[0]),
                        "p": int(G.shape[1]),
                        "fit_seconds": None,
                        "peak_process_rss_gb": None,
                        "internal_oof_cindex": np.nan,
                        "training_cindex": np.nan,
                        "external_cohort": external_cohort,
                        "external_cindex": float(row["c_index"]),
                        "matched_genes": (
                            int(row["n_matched"])
                            if pd.notna(row.get("n_matched"))
                            else F.TOP_K
                        ),
                    }
                    save_json(CACHE_PATH, cache)
                    write_tables(cache, stability_raw)
                elif locked_signature is not None:
                    signature = locked_signature
                    fit_seconds = None
                    peak_rss = None
                    source = "existing submission-lock signature"
                elif scheme == "utility_only":
                    joint_weights, _ = scheme_weights("joint", locked, list(laps))
                    if all(
                        np.isclose(weights[k], joint_weights[k], atol=1e-12)
                        for k in weights
                    ):
                        signature = existing_locked_signature(locked, "joint")
                        fit_seconds = None
                        peak_rss = None
                        source = "identical routing weights to locked joint signature"
                    else:
                        signature, fit_seconds, peak_rss = fit_signature(
                            G, y, L, names
                        )
                        source = "recomputed under utility-only routing"
                else:
                    signature, fit_seconds, peak_rss = fit_signature(G, y, L, names)
                    source = f"recomputed under {scheme} routing"

                if scheme != "stability_only":
                    train_risk, _ = F.frozen_risk(
                        expr_c,
                        signature["genes"],
                        signature["directions"],
                        signature["scores"],
                    )
                    external_risk, matched = F.frozen_risk(
                        external_expr,
                        signature["genes"],
                        signature["directions"],
                        signature["scores"],
                    )
                    if COMPUTE_INTERNAL_OOF:
                        top_idx = np.array(
                            [
                                int(np.where(names == gene)[0][0])
                                for gene in signature["genes"]
                            ]
                        )
                        internal_ci = F.stage2_oof_ci(G[:, top_idx], clin_c, ysurv)
                    else:
                        internal_ci = np.nan
                    cache[cancer][scheme] = {
                        "weights": weights,
                        "no_relation_weight": no_relation,
                        "signature": signature,
                        "signature_source": source,
                        "n": int(G.shape[0]),
                        "p": int(G.shape[1]),
                        "fit_seconds": fit_seconds,
                        "peak_process_rss_gb": peak_rss,
                        "internal_oof_cindex": float(internal_ci),
                        "training_cindex": float(
                            F.evaluate(train_risk, yt, ye)["c_index"]
                        ),
                        "external_cohort": external_cohort,
                        "external_cindex": float(
                            F.evaluate(
                                external_risk, external_time, external_event
                            )["c_index"]
                        ),
                        "matched_genes": int(matched),
                    }
                    save_json(CACHE_PATH, cache)
                    write_tables(cache, stability_raw)

            if scheme not in stability_raw[cancer]:
                reused = None
                if scheme in legacy_raw.get(cancer, {}):
                    reused = legacy_raw[cancer][scheme]
                elif scheme == "utility_only":
                    joint_weights, _ = scheme_weights("joint", locked, list(laps))
                    if all(
                        np.isclose(weights[k], joint_weights[k], atol=1e-12)
                        for k in weights
                    ):
                        reused = (
                            stability_raw[cancer].get("joint")
                            or legacy_raw.get(cancer, {}).get("joint")
                        )
                if reused is not None:
                    stability_raw[cancer][scheme] = reused
                    print("    stability: reused identical locked bootstrap lists", flush=True)
                else:
                    print(f"    stability: computing B={F.N_BOOTSTRAP}", flush=True)
                    stability_raw[cancer][scheme] = {
                        "weights": weights,
                        "top_lists": bootstrap_lists(G, y, L),
                    }
                save_json(STABILITY_RAW_PATH, stability_raw)
                write_tables(cache, stability_raw)

            obj = cache[cancer][scheme]
            print(
                f"    OOF={obj['internal_oof_cindex']:.4f}, "
                f"external={obj['external_cindex']:.4f}",
                flush=True,
            )
            del L
            gc.collect()

        joint_sig = cache[cancer]["joint"]["signature"]
        for scheme in SCHEMES:
            signature = cache[cancer][scheme].get("signature")
            if signature is not None:
                cache[cancer][scheme]["top20_overlap_with_joint"] = len(
                    set(signature["genes"]) & set(joint_sig["genes"])
                )
                cache[cancer][scheme]["signature_identical_to_joint"] = signature_equal(
                    signature, joint_sig
                )
        save_json(CACHE_PATH, cache)
        write_tables(cache, stability_raw)
        del G, y, expr_c, laps
        gc.collect()

    print(f"\nSaved CMPB lock to {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
