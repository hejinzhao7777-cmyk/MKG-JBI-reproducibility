# MKG: reproducibility package

This repository accompanies the manuscript, "MKG: Reliability-gated
multi-omics graph routing for reproducible prognostic signature discovery".
It provides the locked analysis configuration, core source code used for the
principal analyses, manuscript-level source tables, and audit outputs.

Repository URL: <https://github.com/hejinzhao7777-cmyk/MKG-JBI-reproducibility>

## Scope

The study evaluates six TCGA cancer types (LUAD, LIHC, KIRC, COAD, STAD, and HNSC), independent expression cohorts, and a separate end-to-end METABRIC complete-stack audit. The public package contains no participant-level data. TCGA, GEO, and METABRIC are reused public datasets and must be downloaded under their respective terms of use.

## Repository layout

- `code/`: core analysis and audit scripts used for the locked configuration.
- `config/`: locked hyperparameters and dataset manifest.
- `results/`: submission-lock tables, audit outputs, and result JSON files underlying the manuscript.
- `data/`: public-data source index and instructions for placing locally processed inputs.
- `environment.yml`: tested Python dependency specification.

## Quick start

```bash
conda env create -f environment.yml
conda activate mkg
python code/final_config_comparison.py LUAD
```

For full six-cancer reruns, first obtain the public data listed in `data/DATA_SOURCES.md`, preprocess them into the expected local layout, then set `MKG_DATA_ROOT` and `MKG_OUTPUT_ROOT` before running the scripts. The locked results in `results/` are supplied to make every manuscript table auditable without rerunning the computationally intensive pipeline.

```bash
set MKG_DATA_ROOT=D:\\path\\to\\processed_data
set MKG_OUTPUT_ROOT=D:\\path\\to\\mkg_outputs
python code/final_config_comparison.py LUAD LIHC KIRC COAD STAD HNSC
```

The uncertainty audit reconstructs every frozen external score and obtains
patient-bootstrap C-index intervals; it does not refit signatures:

```bash
python code/submission_ci_audit.py
```

The representative leakage audit rebuilds the co-expression, methylation--expression, and CNV graphs inside the training split for LUAD, COAD, and LIHC before routing and Top-20 selection:

```bash
python code/full_train_only_graph_audit.py
```

The latter is computationally intensive. Its bootstrap, random-forest, and PGD settings can be controlled with `FULL_GRAPH_AUDIT_BOOTSTRAP`, `FULL_GRAPH_AUDIT_RF_TREES`, and `FULL_GRAPH_AUDIT_PGD_ITER`; the manuscript-reported run settings are stored with the result JSON.

The METABRIC audit prepares patient-matched expression, promoter methylation,
copy-number, and survival inputs, then reconstructs every graph and selector
inside five prespecified training partitions:

```bash
python code/prepare_metabric_multiomics.py --help
python code/metabric_multiomics_portability_audit.py --help
```

The public result files include input hashes, split-level routing utilities,
selected weights, Top-20 signatures, and untouched-test C-indices. The
participant-level processed matrices remain local and are not included.

The repeated strict audit uses three prespecified train/test splits per cancer
and reconstructs all relation graphs within each training split:

```bash
python code/cmpb_repeated_train_only_graph_audit.py --cancer LUAD --split-seed 42
```

The right-censored gate-margin sensitivity can be rerun independently:

```bash
python code/cmpb_conservative_gate_sensitivity.py
```

The complete six-method stability figure and paired MKG--Uni-Cox audit are
generated from the supplied cancer-level source table:

```bash
python code/cmpb_full_stability_baseline_figure.py \
  --input results/source_tables/TableS_expanded_stability_baselines_jbi.csv \
  --outdir results/full_stability_baseline
```

See `REPRODUCIBILITY.md` for the evidence layers, exact aggregation command,
and interpretation boundaries.

The completed repeated audit is stored in
`results/repeated_train_only_graph_audit/`, including all 18 run records,
SHA-256 hashes, cancer-clustered summaries, and the submission figure.

## Locked configuration

The submission lock uses `lambda1=0.2`, `lambda2=50`, `gamma=10`, Top-20
signatures, 30 bootstrap resamples, normalized truncated RBO@20 (`p=0.9`),
and a zero-Laplacian no-relation baseline. Full provenance, output hashes,
and step exit codes are recorded in the locked manifest. Historical
filenames containing `JBI` identify the original computational lock and do
not indicate a journal-specific algorithm.

## Data and code availability

The manuscript-ready availability wording is in `data/AVAILABILITY_STATEMENT.md`. Do not claim that an archival DOI exists until a DOI-minting release has actually been created.

## Citation

Please cite the associated manuscript after publication. Repository release and archival DOI information will be added here at publication.

## License

Code is released under the MIT License. Reused public datasets remain subject to the terms of their original repositories.
