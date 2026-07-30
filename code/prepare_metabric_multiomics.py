"""Prepare a locked-gene, patient-aligned METABRIC multi-omics survival cohort.

The script deliberately uses no outcome-driven feature filtering.  Its gene
universe is the intersection of the expression feature sets used by all six
locked TCGA analyses, followed by availability in all three METABRIC assays.
Public study matrices are streamed from the cBioPortal Datahub Git LFS store;
clinical outcomes are obtained from the public cBioPortal API.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests


STUDY = "brca_metabric"
DATAHUB_BASE = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/"
    "master/public/brca_metabric"
)
CBIO_API = "https://www.cbioportal.org/api"
MATRIX_SPECS = {
    "expression": {
        "filename": "data_mrna_illumina_microarray.txt",
        "metadata_columns": 2,
        "output": "expr_final.tsv",
    },
    "methylation": {
        "filename": "data_methylation_promoters_rrbs.txt",
        "metadata_columns": 1,
        "output": "meth_gene_level.tsv",
    },
    "copy_number": {
        "filename": "data_cna.txt",
        "metadata_columns": 2,
        "output": "cnv_aligned.tsv",
    },
}
LOCKED_CANCERS = ("LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC")


def request_with_retry(
    url: str, *, stream: bool = False, params: dict | None = None, attempts: int = 5
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                params=params,
                stream=stream,
                timeout=(30, 300),
                headers={"User-Agent": "MKG-reproducibility-audit/1.0"},
            )
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not retrieve {url}") from last_error


def read_matrix_header(
    filename: str, metadata_columns: int, cache_dir: Path | None
) -> list[str]:
    if cache_dir is not None:
        path = cache_dir / filename
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.readline().rstrip("\r\n").split("\t")[metadata_columns:]
    url = f"{DATAHUB_BASE}/{filename}"
    response = request_with_retry(url, stream=True)
    try:
        response.raw.decode_content = True
        wrapper = io.TextIOWrapper(response.raw, encoding="utf-8", newline="")
        header = wrapper.readline().rstrip("\r\n").split("\t")
        return header[metadata_columns:]
    finally:
        response.close()


def locked_gene_universe(data_root: Path) -> set[str]:
    sets: list[set[str]] = []
    for cancer in LOCKED_CANCERS:
        path = data_root / cancer / "expr_final.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Missing locked expression matrix: {path}")
        columns = pd.read_csv(path, sep="\t", nrows=0, index_col=0).columns
        sets.append({str(value).strip() for value in columns if str(value).strip()})
    return set.intersection(*sets)


def clinical_survival() -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    response = request_with_retry(
        f"{CBIO_API}/studies/{STUDY}/clinical-data",
        params={
            "clinicalDataType": "PATIENT",
            "projection": "DETAILED",
            "pageSize": 100000,
        },
    )
    records = response.json()
    by_patient: dict[str, dict[str, str]] = defaultdict(dict)
    for record in records:
        by_patient[str(record["patientId"])][
            str(record["clinicalAttributeId"])
        ] = str(record["value"])

    rows = []
    for patient, attributes in by_patient.items():
        raw_time = attributes.get("OS_MONTHS")
        raw_status = attributes.get("OS_STATUS")
        try:
            time_months = float(raw_time) if raw_time is not None else np.nan
        except ValueError:
            time_months = np.nan
        if raw_status is None:
            event = np.nan
        else:
            event = int(
                raw_status.startswith("1")
                or "DECEASED" in raw_status.upper()
                or raw_status.upper() == "DEAD"
            )
        if np.isfinite(time_months) and time_months > 0 and np.isfinite(event):
            rows.append((patient, time_months * 30.4375, int(event)))
    frame = pd.DataFrame(rows, columns=["sample_id", "OS_time", "OS"]).set_index(
        "sample_id"
    )
    return frame.sort_index(), by_patient


def parse_value(value: str) -> float:
    if value in {"", "NA", "NaN", "nan", "null", "None"}:
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def stream_selected_matrix(
    filename: str,
    metadata_columns: int,
    selected_samples: list[str],
    selected_genes: set[str],
    cache_dir: Path | None,
) -> tuple[pd.DataFrame, dict]:
    url = f"{DATAHUB_BASE}/{filename}"
    values_by_gene: dict[str, list[np.ndarray]] = defaultdict(list)
    total_rows = 0
    selected_rows = 0
    response: requests.Response | None = None
    if cache_dir is not None:
        local_path = cache_dir / filename
        wrapper = local_path.open("r", encoding="utf-8", newline="")
    else:
        response = request_with_retry(url, stream=True)
        response.raw.decode_content = True
        wrapper = io.TextIOWrapper(response.raw, encoding="utf-8", newline="")
    try:
        header = wrapper.readline().rstrip("\r\n").split("\t")
        sample_columns = header[metadata_columns:]
        sample_position = {sample: index for index, sample in enumerate(sample_columns)}
        missing_samples = [
            sample for sample in selected_samples if sample not in sample_position
        ]
        if missing_samples:
            raise ValueError(
                f"{filename} lacks {len(missing_samples)} selected samples; "
                f"examples: {missing_samples[:5]}"
            )
        positions = np.asarray(
            [sample_position[sample] + metadata_columns for sample in selected_samples],
            dtype=int,
        )
        max_position = int(positions.max())
        for line in wrapper:
            total_rows += 1
            gene = line.split("\t", 1)[0].strip()
            if gene not in selected_genes:
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) <= max_position:
                continue
            vector = np.fromiter(
                (parse_value(fields[position]) for position in positions),
                dtype=np.float32,
                count=len(positions),
            )
            values_by_gene[gene].append(vector)
            selected_rows += 1
    finally:
        wrapper.close()
        if response is not None:
            response.close()

    collapsed: dict[str, np.ndarray] = {}
    duplicate_genes = 0
    for gene, vectors in values_by_gene.items():
        if len(vectors) == 1:
            collapsed[gene] = vectors[0]
        else:
            duplicate_genes += 1
            collapsed[gene] = np.nanmean(np.vstack(vectors), axis=0).astype(np.float32)
    frame = pd.DataFrame.from_dict(
        collapsed, orient="index", columns=selected_samples, dtype=np.float32
    ).T
    frame.index.name = "sample_id"
    metadata = {
        "url": url,
        "local_cache": str((cache_dir / filename).resolve())
        if cache_dir is not None
        else None,
        "remote_rows_scanned": total_rows,
        "selected_rows_before_duplicate_collapse": selected_rows,
        "selected_genes": int(frame.shape[1]),
        "selected_samples": int(frame.shape[0]),
        "duplicate_gene_symbols_collapsed_by_mean": duplicate_genes,
        "missing_fraction": float(frame.isna().to_numpy().mean()),
    }
    return frame, metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = (args.output or data_root / "METABRIC").resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve() if args.cache_dir else None
    if cache_dir is not None:
        for spec in MATRIX_SPECS.values():
            path = cache_dir / spec["filename"]
            if not path.exists() or path.stat().st_size < 1_000_000:
                raise FileNotFoundError(f"Missing or incomplete cache file: {path}")

    locked_genes = locked_gene_universe(data_root)
    headers = {
        assay: read_matrix_header(
            spec["filename"], spec["metadata_columns"], cache_dir
        )
        for assay, spec in MATRIX_SPECS.items()
    }
    assay_sample_overlap = set.intersection(*(set(values) for values in headers.values()))
    survival, clinical_attributes = clinical_survival()
    selected_samples = sorted(assay_sample_overlap & set(survival.index))

    matrices: dict[str, pd.DataFrame] = {}
    matrix_metadata: dict[str, dict] = {}
    for assay, spec in MATRIX_SPECS.items():
        print(f"[METABRIC] streaming {assay}", flush=True)
        frame, metadata = stream_selected_matrix(
            spec["filename"],
            spec["metadata_columns"],
            selected_samples,
            locked_genes,
            cache_dir,
        )
        matrices[assay] = frame
        matrix_metadata[assay] = metadata

    common_genes = sorted(
        locked_genes
        & set(matrices["expression"].columns)
        & set(matrices["methylation"].columns)
        & set(matrices["copy_number"].columns)
    )
    if len(common_genes) < 1000:
        raise RuntimeError(f"Only {len(common_genes)} common genes remained")

    output_files: dict[str, str] = {}
    for assay, spec in MATRIX_SPECS.items():
        path = output / spec["output"]
        matrices[assay].loc[selected_samples, common_genes].to_csv(path, sep="\t")
        output_files[assay] = path.name
    survival_path = output / "deviance_residuals.tsv"
    survival.loc[selected_samples, ["OS_time", "OS"]].to_csv(survival_path, sep="\t")
    clinical_path = output / "clinical_covariates.tsv"
    clinical_rows = []
    for sample in selected_samples:
        attributes = clinical_attributes.get(sample, {})
        clinical_rows.append(
            {
                "sample_id": sample,
                "age": pd.to_numeric(
                    attributes.get("AGE_AT_DIAGNOSIS"), errors="coerce"
                ),
                "gender": attributes.get("SEX"),
            }
        )
    clinical_frame = pd.DataFrame(clinical_rows).set_index("sample_id")
    clinical_frame.to_csv(clinical_path, sep="\t")

    manifest = {
        "study": STUDY,
        "design_role": "independent end-to-end multi-omics portability audit",
        "selection_rule": (
            "intersection of the six locked TCGA expression universes and genes "
            "available in all three METABRIC assays; no outcome-driven filtering"
        ),
        "source": {
            "cBioPortal_api": f"{CBIO_API}/studies/{STUDY}",
            "cBioPortal_datahub": (
                "https://github.com/cBioPortal/datahub/tree/master/public/brca_metabric"
            ),
        },
        "raw_cache_sha256": {
            spec["filename"]: sha256(cache_dir / spec["filename"])
            for spec in MATRIX_SPECS.values()
        }
        if cache_dir is not None
        else None,
        "locked_six_cancer_gene_intersection": len(locked_genes),
        "assay_sample_overlap_before_survival_filter": len(assay_sample_overlap),
        "analysis_samples": len(selected_samples),
        "events": int(survival.loc[selected_samples, "OS"].sum()),
        "censored": int(
            len(selected_samples) - survival.loc[selected_samples, "OS"].sum()
        ),
        "clinical_covariates": {
            "age_nonmissing": int(clinical_frame["age"].notna().sum()),
            "gender_nonmissing": int(clinical_frame["gender"].notna().sum()),
            "stage": "not available in the public patient-level study table",
        },
        "analysis_genes": len(common_genes),
        "matrix_metadata": matrix_metadata,
        "files": {},
    }
    for label, filename in {
        **output_files,
        "survival": survival_path.name,
        "clinical": clinical_path.name,
    }.items():
        path = output / filename
        manifest["files"][label] = {
            "filename": filename,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    (output / "METABRIC_PREPARATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
