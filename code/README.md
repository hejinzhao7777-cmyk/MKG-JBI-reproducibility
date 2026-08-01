# Analysis scripts

The scripts in this folder are the source files used for the locked analyses. `final_config_comparison.py` is configured for portable paths:

```bash
set MKG_DATA_ROOT=D:\\path\\to\\processed_data
set MKG_OUTPUT_ROOT=D:\\path\\to\\mkg_outputs
python final_config_comparison.py LUAD
```

The remaining scripts preserve the locked submission workflow and its specific audit calculations. Some audit scripts were executed from the original project layout; before an end-to-end rerun, review their path constants and point them to the same `MKG_DATA_ROOT` and `MKG_OUTPUT_ROOT` locations. This explicit note is intentional: it prevents a misleading claim that the public release is turnkey before the full preprocessing pipeline and distributable processed matrices have been deposited.

`results/` contains the locked outputs used by the manuscript, so numerical claims can be inspected without rerunning the compute-intensive workflow.

Submission-facing audits:

- `run_cmpb_lock_pipeline.py` is the CMPB-named orchestration entry point for
  consolidating the six-cancer lock and writing its manifest in the original
  analysis layout.
- `submission_ci_audit.py` reconstructs locked external molecular scores and estimates patient- and cancer-bootstrap confidence intervals.
- `full_train_only_graph_audit.py` reconstructs all three graph layers within representative training splits before routing, selection, and held-out evaluation.
- `cmpb_five_arm_ablation.py` runs the locked no-graph, equal-weight,
  stability-only, utility-only, and joint-routing comparison. It accepts one
  or more cancer codes as command-line arguments and saves each cancer
  incrementally.
- `cmpb_synthetic_component_ablation.py` evaluates the same five routing arms
  under controlled reliable, complementary, adversarial, harmful, and
  unstable-relation scenarios using an untouched test split.
- `prepare_metabric_multiomics.py` aligns public METABRIC expression,
  promoter-methylation, copy-number, clinical, and survival profiles to the
  outcome-free locked gene universe and writes a SHA-256 preparation manifest.
- `metabric_multiomics_portability_audit.py` performs five prespecified
  train-only complete-stack reconstructions and writes split-level routing,
  signature-stability, and untouched-test results. Use `--data` to point to
  the locally prepared METABRIC directory; run `--help` for the full
  interface. The preparation script accepts `--data-root`, `--output`, and an
  optional `--cache-dir`.
- `cmpb_conservative_gate_sensitivity.py` varies the minimum validation
  C-index gain required for graph eligibility in the right-censored stress
  test.
- `cmpb_repeated_train_only_graph_audit.py` rebuilds every graph inside one
  prespecified training split for one cancer and records fixed-versus-
  reconstructed contrasts at three gate margins.
- `assemble_repeated_train_only_graph_audit.py` combines completed
  cancer-by-split audits and computes cancer-clustered intervals.
- `cmpb_conditional_fused_stability.py` refits both MKG ranking engines and
  five comparators on the same 30 independent 80% subsamples while freezing
  only the complete-cohort MKG route and fusion weights.
- `cmpb_cv_tuned_cox_baselines.py` selects Cox-Lasso and Cox elastic-net
  penalties by five-fold development-only cross-validation before external
  transfer; `cmpb_tuned_cox_patient_bootstrap.py` adds patient-bootstrap
  uncertainty for their frozen external scores.
- `cmpb_additional_external_cohorts.py` reports the smaller within-cancer
  cohorts already present in the external lock, without double-weighting
  those cancers in primary summaries.
- `cmpb_build_final_external_source_table.py` combines the locked-score,
  cross-validated sparse-Cox, and additional-cohort patient-bootstrap outputs
  into the final eight-cohort comparison table.
- `cmpb_directional_methylation_graph_audit.py` rebuilds all six methylation
  graphs using both ordered directions; `cmpb_directional_methylation_routing_audit.py`
  reruns the formal route with only that layer changed.
- `cmpb_pgd_convergence_audit.py`, `cmpb_fista_component_audit.py`, and
  `cmpb_fista_routing_sensitivity.py` compare the 300-update lock with a
  high-accuracy accelerated proximal-gradient reference.
- `cmpb_finalize_all6_sensitivity_audits.py` merges split-worker FISTA and
  directional-graph reruns and adds matched primary-score contrasts.
- `cmpb_full_stability_baseline_figure.py` regenerates the complete
  six-method cancer-level stability comparison and its exact paired
  cancer-cluster bootstrap intervals against Uni-Cox and cross-validated
  Cox elastic net.
- `cmpb_combine_stability_results.py` joins the conditional MKG/comparator
  stability output with the development-CV-tuned sparse-Cox stability output
  before the final figure is rendered.
- `cmpb_merge_and_plot.py` merges the completed five-arm workers, recomputes
  cancer-bootstrap summaries, and renders the submission figure with every
  cancer displayed directly.
- `conformal_ipcw_cmpb.py` runs the censoring-aware conformal calibration
  audit and writes CMPB-named JSON, CSV, and figure outputs.
- `routing_reliability_simulation.py` regenerates the two-panel routing
  reliability stress-test figure and its aggregate CSV/JSON outputs.
- `make_figures.py` contains the source-table-based legacy figure builders;
  call `fig6()` to reproduce the final hyperparameter-sensitivity panel.
- `make_fig9.py` regenerates the submission-lock ablation and external
  decision-curve panel. The two figure scripts accept `MKG_RESULTS_DIR` and
  `MKG_FIGURE_DIR` environment variables for portable input/output paths.
- `make_cmpb_graphical_abstract.py` regenerates the simplified four-stage
  vector graphical abstract and its PDF, SVG, PNG, and TIFF exports.
- `cmpb_final_submission_qa.py` checks word limits, citations, referenced
  figures, highlights, stale terminology, and final LaTeX logs.
- `verify_cmpb_release.py` scans every tracked filename and supported text,
  JSON, CSV, PDF, PNG, and SVG file for naming residue and structural damage.

The sensitivity, decision-curve, routing-simulation, repeated-audit,
complete-stability, and five-arm scripts generate their own source tables or
figures. Numerical source tables are supplied for the remaining historical
plots whose original figure-generation scripts remain outside the public
release.
