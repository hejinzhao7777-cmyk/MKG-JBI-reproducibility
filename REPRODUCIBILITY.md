
# Reproducibility contract

This repository distinguishes evidence layers so that manuscript claims can be
audited without conflating reruns, locked outputs, and independent validation.

## 1. Locked manuscript outputs

`results/submission_lock/` and `results/source_tables/` contain the numerical
outputs used by the CMPB manuscript. Filenames and internal references follow
one CMPB naming scheme. The naming migration did not change any numeric table,
model output, cohort definition, or locked hyperparameter.

## 2. Mechanism and sensitivity audits

`results/component_ablation/` contains the five-arm real-data and controlled
component experiments. `results/conservative_gate_sensitivity/` tests fixed
minimum validation-gain margins under right-censored simulation. The latter is
a boundary analysis: stricter margins increased no-graph activation but did
not eliminate finite-sample false admission in the all-harmful scenario.

## 3. Leakage audits

The strict audit reconstructs expression scaling and all three relation graphs
inside a training split before routing, Top-20 selection, and held-out
evaluation. Run one cancer and one split at a time:

```bash
set MKG_DATA_ROOT=D:\path\to\processed_data
set MKG_OUTPUT_ROOT=D:\path\to\mkg_outputs
python code/cmpb_repeated_train_only_graph_audit.py --cancer LUAD --split-seed 42
```

The manuscript audit uses split seeds 42, 2025, and 7301 for each of LUAD,
LIHC, KIRC, COAD, STAD, and HNSC. Aggregate completed runs with:

```bash
python code/assemble_repeated_train_only_graph_audit.py ^
  --root D:\path\to\mkg_outputs\repeated_train_only_graph_audit ^
  --outdir D:\path\to\mkg_outputs\repeated_train_only_graph_audit\summary
```

The aggregate confidence interval resamples six cancer-specific split means.
It does not treat the 18 cancer-by-split rows as independent cohorts. Each
strict run uses 10 stability resamples, 100 stage-1 random-forest trees, and
an explicit 300-iteration proximal-gradient cap.

All 18 prespecified runs are supplied under
`results/repeated_train_only_graph_audit/runs/`. At the primary zero margin,
the mean fixed-minus-reconstructed held-out C-index was -0.001505
(cancer-clustered 95% interval -0.009619 to 0.005814), mean Top-20 Jaccard was
0.841259, and route modes agreed in 13/18 audits. The small mean contrast does
not erase the five split-specific routing changes.

## 4. Independent METABRIC complete-stack audit

The preparation script aligns public METABRIC expression, promoter
methylation, copy-number, survival, and clinical profiles without
outcome-driven feature filtering. The portability script rebuilds all three
graphs, routing, Top-20 selection, and the reduced Cox model inside each of
five prespecified training partitions:

```bash
python code/prepare_metabric_multiomics.py --help
python code/metabric_multiomics_portability_audit.py --help
```

The final audit contained 1,416 patients, 830 deaths, and 2,230 aligned genes.
Four of five splits rejected all graph layers. Mean held-out MKG-minus-zero
C-index was 0.0021. This evidence supports operational abstention on another
complete multi-omics stack, not a general performance advantage or
same-cancer frozen-signature replication.

At the manuscript's 300-tree stage-1 budget, representative seed 42 remained
reject-all. Boundary seed 71 admitted methylation and CNA and had a held-out
MKG-minus-zero difference of -0.0169. This computation-budget sensitivity is
reported as a finite-sample harmful admission: a positive development routing
score is a safeguard, not a guarantee.

Participant-level METABRIC matrices are not redistributed. The public package
contains preparation hashes, split-level aggregate outputs, and the full
cohort-screening record.

## Controlled gate sensitivity

```bash
set MKG_OUTPUT_ROOT=D:\path\to\mkg_outputs
python code/cmpb_conservative_gate_sensitivity.py
```

The script writes row-level results before summaries and figures. Hidden signal
support is recorded only for diagnostic evaluation and is never used by the
routing rule.

## Complete stability comparator audit

The full cancer-level comparator table, rather than a selected Cox-Lasso
contrast, is the source for the main stability figure:

```bash
python code/cmpb_full_stability_baseline_figure.py \
  --input results/source_tables/TableS_expanded_stability_baselines_cmpb.csv \
  --outdir results/full_stability_baseline
```

Every one of the six cancer values is displayed. The paired MKG--Uni-Cox and
MKG--cross-validated-Cox-elastic-net intervals exhaustively enumerate all
\(6^6\) empirical cancer-cluster bootstrap resamples. The result supports
competitive stability, not uniform superiority over the strongest comparator.

## Final methodological audits

`results/final_audits/` contains the final pre-submission checks. The supplied
scripts and result records distinguish four questions that should not be
conflated:

- conditional feature-ranking stability after freezing the complete-cohort
  MKG route and fusion weights;
- external discrimination of Cox-Lasso and Cox elastic net after five-fold
  development-only penalty selection;
- downstream sensitivity to an order-invariant directional-maximum
  methylation graph; and
- sensitivity of the fixed 300-update proximal-gradient feature generator to
  a converged accelerated proximal-gradient reference.

The 300-update output is an explicit algorithmic lock, not a claim of exact
objective minimization. The accelerated-reference audits retain the locked
data splits, graph candidates, stability scores, and evaluation protocol so
that numerical optimization sensitivity can be inspected separately from
sampling and routing sensitivity.

The final audits were executed with Python 3.13.5, NumPy 2.3.4, pandas 2.3.3,
SciPy 1.16.2, scikit-learn 1.8.0, scikit-survival 0.27.0, lifelines 0.30.3,
Matplotlib 3.10.7, and PyTorch 2.11.0+cu128. `environment.yml` gives the
portable dependency specification; this version record identifies the tested
workstation runtime.

## Data boundary

Participant-level matrices are not redistributed here. Obtain TCGA, GEO, and
METABRIC data from the repositories listed in `data/DATA_SOURCES.md`, comply
with their terms, and place processed inputs under a local data root. External
outcomes must remain isolated until the frozen molecular score is evaluated.

## Interpretation boundary

MKG is evaluated as a compact signature-selection and frozen-score workflow.
Attribution-derived RSF and DeepSurv rows are not native retrained predictors.
Selection stability, external discrimination, and graph-route reproducibility
are separate endpoints.
