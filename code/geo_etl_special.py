"""
经典强队列专用 ETL（生存数据不在 series matrix 内）：
  GSE14520 (LIHC) —— 生存在 GEO 补充文件 GSE14520_Extra_Supplement.txt
  E-MTAB-1980 (KIRC) —— ArrayExpress
用法: python geo_etl_special.py GSE14520
"""
import os
import sys, io, gzip, urllib.request
import numpy as np
import pandas as pd
import GEOparse
from pathlib import Path

RAW = Path(os.environ.get("MKG_RAW_ROOT", "data/raw"))
PROC = Path(os.environ.get("MKG_DATA_ROOT", "data/processed"))
RAW.mkdir(parents=True, exist_ok=True)


def pick_symbol_col(t):
    for c in ["Gene Symbol", "Symbol", "GENE_SYMBOL", "gene_symbol", "ILMN_Gene"]:
        if c in t.columns:
            return c
    for c in t.columns:
        if "symbol" in c.lower():
            return c
    raise RuntimeError(f"无符号列: {list(t.columns)}")


def collapse(expr_probes, probe2gene):
    expr_probes.index = expr_probes.index.astype(str)
    expr_probes["__g__"] = [probe2gene.get(p) for p in expr_probes.index]
    expr_probes = expr_probes.dropna(subset=["__g__"])
    expr_probes = expr_probes[expr_probes["__g__"] != ""]
    expr_probes["__m__"] = expr_probes.drop(columns="__g__").mean(axis=1, numeric_only=True)
    expr_probes = expr_probes.sort_values("__m__", ascending=False).drop_duplicates("__g__")
    return expr_probes.set_index("__g__").drop(columns="__m__").apply(pd.to_numeric, errors="coerce")


def etl_gse14520():
    # 1. 生存（补充文件）
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/suppl/GSE14520_Extra_Supplement.txt.gz"
    txt = gzip.decompress(urllib.request.urlopen(url, timeout=180).read()).decode("utf-8", "replace")
    sup = pd.read_csv(io.StringIO(txt), sep="\t")
    sup = sup[sup["Tissue Type"].astype(str).str.strip().str.lower() == "tumor"].copy()
    sup["sample"] = sup["Affy_GSM"].astype(str)
    sup["OS"] = pd.to_numeric(sup["Survival status"], errors="coerce")
    sup["OS.time"] = pd.to_numeric(sup["Survival months"], errors="coerce") * 30.44
    surv = sup[["sample", "OS.time", "OS"]].dropna()
    surv = surv[(surv["OS.time"] > 0) & surv["OS"].isin([0, 1])]
    print(f"  生存: {len(surv)} 肿瘤样本, 事件率={surv['OS'].mean():.1%}")

    # 2. 表达（GPL3921 主平台）
    gse = GEOparse.get_GEO(geo="GSE14520", destdir=str(RAW), silent=True)
    gpl = gse.gpls.get("GPL3921") or list(gse.gpls.values())[0]
    sym = pick_symbol_col(gpl.table)
    probe2gene = {str(k): (str(v).split("///")[0].strip() if pd.notna(v) else None)
                  for k, v in gpl.table.set_index("ID")[sym].items()}
    # 只取属于 GPL3921 的样本
    gsms3921 = [g for g, s in gse.gsms.items() if s.metadata.get("platform_id", [""])[0] == "GPL3921"]
    sub = {g: gse.gsms[g] for g in gsms3921}
    expr = pd.DataFrame({g: s.table.set_index("ID_REF")["VALUE"] for g, s in sub.items()})
    expr = collapse(expr, probe2gene).T
    print(f"  表达: {expr.shape[0]} 样本 × {expr.shape[1]} 基因")

    common = sorted(set(expr.index) & set(surv["sample"]))
    out = PROC / "LIHC" / "external" / "GSE14520"
    out.mkdir(parents=True, exist_ok=True)
    e = expr.loc[common]; e.index.name = "sample_id"
    e.to_csv(out / "expr.tsv", sep="\t")
    surv.set_index("sample").loc[common].reset_index().to_csv(out / "survival.tsv", sep="\t", index=False)
    print(f"  保存: {out}  (对齐 {len(common)} 样本)")


def main():
    {"GSE14520": etl_gse14520}[sys.argv[1]]()


if __name__ == "__main__":
    main()

