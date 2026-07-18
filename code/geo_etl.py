"""
GEO 外部队列 ETL：下载 → 探针映射基因 → 解析生存 → 输出 expr.tsv / survival.tsv
配置驱动，逐数据集添加。用法: python geo_etl.py GSE65858
"""
import os
import sys, re
import numpy as np
import pandas as pd
import GEOparse
from pathlib import Path

RAW = Path(os.environ.get("MKG_RAW_ROOT", "data/raw"))
PROC = Path(os.environ.get("MKG_DATA_ROOT", "data/processed"))
RAW.mkdir(parents=True, exist_ok=True)

# 每数据集配置：癌种 / 生存时间键 / 事件键 / 事件映射 / 时间单位
CFG = {
    "GSE65858": {"cancer": "HNSC", "os_key": "os", "event_key": "os_event",
                 "event_map": {"true": 1, "false": 0, "1": 1, "0": 0, "dead": 1, "alive": 0},
                 "time_unit": "day"},
    "GSE39582": {"cancer": "COAD", "os_key": "os.delay (months)", "event_key": "os.event",
                 "event_map": {"1": 1, "0": 0, "true": 1, "false": 0},
                 "time_unit": "month"},
    "GSE84437": {"cancer": "STAD", "os_key": "duration overall survival", "event_key": "death",
                 "event_map": {"1": 1, "0": 0, "dead": 1, "alive": 0},
                 "time_unit": "month"},
    "GSE76427": {"cancer": "LIHC", "os_key": "duryears_os", "event_key": "event_os",
                 "event_map": {"1": 1, "0": 0},
                 "time_unit": "year"},
    "GSE29609": {"cancer": "KIRC", "os_key": "survival time", "event_key": "death (1=yes, 0=no)",
                 "event_map": {"1": 1, "0": 0},
                 "time_unit": "month"},
}


def pick_symbol_col(gpl_table):
    cands = ["Symbol", "ILMN_Gene", "Gene Symbol", "GENE_SYMBOL", "gene_symbol",
             "GeneSymbol", "Gene_Symbol", "SYMBOL", "gene_assignment"]
    for c in cands:
        if c in gpl_table.columns:
            return c
    for c in gpl_table.columns:
        if "symbol" in c.lower():
            return c
    raise RuntimeError(f"找不到基因符号列；可用列: {list(gpl_table.columns)}")


def parse_symbol(val):
    if pd.isna(val):
        return None
    s = str(val)
    if "//" in s:                       # gene_assignment 形如 "NM_x // SYMBOL // desc"
        parts = [p.strip() for p in s.split("//")]
        return parts[1] if len(parts) > 1 else None
    return s.strip().split(" ")[0] if s.strip() else None


def get_char(gsm, key):
    for c in gsm.metadata.get("characteristics_ch1", []):
        if c.split(":")[0].strip().lower() == key.lower():
            return c.split(":", 1)[1].strip()
    return None


def main():
    gid = sys.argv[1]
    cfg = CFG[gid]
    gse = GEOparse.get_GEO(geo=gid, destdir=str(RAW), silent=True)
    gpl = list(gse.gpls.values())[0]
    sym_col = pick_symbol_col(gpl.table)
    print(f"{gid}: 平台符号列 = {sym_col}")

    # 探针 → 基因符号
    ann = gpl.table.set_index("ID")[sym_col]
    probe2gene = {str(k): parse_symbol(v) for k, v in ann.items()}

    # 表达矩阵 (探针 × 样本)
    expr = gse.pivot_samples("VALUE")
    expr.index = expr.index.astype(str)
    expr["__gene__"] = [probe2gene.get(p) for p in expr.index]
    expr = expr.dropna(subset=["__gene__"])
    expr = expr[expr["__gene__"] != ""]
    # 一基因多探针 → 取平均表达最高的探针
    expr["__m__"] = expr.drop(columns="__gene__").mean(axis=1, numeric_only=True)
    expr = expr.sort_values("__m__", ascending=False).drop_duplicates("__gene__")
    expr = expr.set_index("__gene__").drop(columns="__m__")
    expr = expr.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    expr_samples = expr.T                      # 样本 × 基因
    print(f"  表达矩阵: {expr_samples.shape[0]} 样本 × {expr_samples.shape[1]} 基因")

    # 生存
    rows = []
    for gsm_id, gsm in gse.gsms.items():
        t = get_char(gsm, cfg["os_key"])
        e = get_char(gsm, cfg["event_key"])
        if t is None or e is None:
            continue
        try:
            tt = float(t)
        except Exception:
            continue
        ev = cfg["event_map"].get(str(e).strip().lower())
        if ev is None:
            try:
                ev = int(float(e))
            except Exception:
                continue
        if cfg["time_unit"] == "month":
            tt *= 30.44
        elif cfg["time_unit"] == "year":
            tt *= 365.25
        rows.append({"sample": gsm_id, "OS.time": tt, "OS": int(ev)})
    surv = pd.DataFrame(rows)
    surv = surv[(surv["OS.time"] > 0) & surv["OS"].isin([0, 1])]
    print(f"  生存: {len(surv)} 样本, 事件率={surv['OS'].mean():.1%}, "
          f"OS.time 天数 中位={surv['OS.time'].median():.0f} 最大={surv['OS.time'].max():.0f}")

    # 对齐 + 保存
    common = sorted(set(expr_samples.index) & set(surv["sample"]))
    expr_out = expr_samples.loc[common]
    surv_out = surv.set_index("sample").loc[common].reset_index()
    out = PROC / cfg["cancer"] / "external" / gid
    out.mkdir(parents=True, exist_ok=True)
    expr_out.index.name = "sample_id"
    expr_out.to_csv(out / "expr.tsv", sep="\t")
    surv_out.to_csv(out / "survival.tsv", sep="\t", index=False)
    print(f"  保存: {out}  (对齐 {len(common)} 样本)")


if __name__ == "__main__":
    main()

