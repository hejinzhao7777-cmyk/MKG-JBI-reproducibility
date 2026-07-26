# MKG: reproducibility package for the JBI submission

This repository accompanies the manuscript, "MKG: stability-utility gated graph routing for reliable multi-omics survival signatures". It provides the locked analysis configuration, the source code used for the principal analyses, the manuscript-level source tables, and the resulting audit files.

Repository URL: <https://github.com/hejinzhao7777-cmyk/MKG-JBI-reproducibility>

## Scope

The study evaluates six TCGA cancer types (LUAD, LIHC, KIRC, COAD, STAD, and HNSC) with independent external cohorts. The public package contains no participant-level data. TCGA and GEO are reused public datasets and must be downloaded under their respective terms of use.

## Repository layout

- `code/`: analysis and audit scripts used for the locked JBI configuration.
- `config/`: locked hyperparameters and dataset manifest.
- `results/`: submission-lock tables, audit outputs, and result JSON files underlying the manuscript.
- `data/`: public-data source index and instructions for placing locally processed inputs.
- `environment.yml`: tested Python dependency specification.

## Quick start

```bash
conda env create -f environment.yml
conda activate mkg-jbi
python code/final_config_comparison.py LUAD
```

For full six-cancer reruns, first obtain the public data listed in `data/DATA_SOURCES.md`, preprocess them into the expected local layout, then set `MKG_DATA_ROOT` and `MKG_OUTPUT_ROOT` before running the scripts. The locked results in `results/` are supplied to make every manuscript table auditable without rerunning the computationally intensive pipeline.

```bash
set MKG_DATA_ROOT=D:\\path\\to\\processed_data
set MKG_OUTPUT_ROOT=D:\\path\\to\\mkg_outputs
python code/final_config_comparison.py LUAD LIHC KIRC COAD STAD HNSC
```

The submission-level uncertainty audit reconstructs every frozen external score and obtains patient-bootstrap C-index intervals; it does not refit signatures:

```bash
python code/submission_ci_audit.py
```

The representative leakage audit rebuilds the co-expression, methylation--expression, and CNV graphs inside the training split for LUAD, COAD, and LIHC before routing and Top-20 selection:

```bash
python code/full_train_only_graph_audit.py
```

The latter is computationally intensive. Its bootstrap, random-forest, and PGD settings can be controlled with `FULL_GRAPH_AUDIT_BOOTSTRAP`, `FULL_GRAPH_AUDIT_RF_TREES`, and `FULL_GRAPH_AUDIT_PGD_ITER`; the manuscript-reported run settings are stored with the result JSON.

## Locked configuration

The submission lock uses `lambda1=0.2`, `lambda2=50`, `gamma=10`, Top-20 signatures, 30 bootstrap resamples, normalized truncated RBO@20 (`p=0.9`), and a zero-Laplacian no-relation baseline. Full provenance, output hashes, and step exit codes are recorded in `config/MKG_JBI_SUBMISSION_LOCK_manifest.json`.

## Data and code availability

The manuscript-ready availability wording is in `data/AVAILABILITY_STATEMENT.md`. Do not claim that an archival DOI exists until a DOI-minting release has actually been created.

## Citation

Please cite the associated manuscript after publication. Repository release and archival DOI information will be added here at publication.

## License

Code is released under the MIT License. Reused public datasets remain subject to the terms of their original repositories.

