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

- `submission_ci_audit.py` reconstructs locked external molecular scores and estimates patient- and cancer-bootstrap confidence intervals.
- `full_train_only_graph_audit.py` reconstructs all three graph layers within representative training splits before routing, selection, and held-out evaluation.
- `cmpb_five_arm_ablation.py` runs the locked no-graph, equal-weight,
  stability-only, utility-only, and joint-routing comparison. It accepts one
  or more cancer codes as command-line arguments and saves each cancer
  incrementally.
- `cmpb_synthetic_component_ablation.py` evaluates the same five routing arms
  under controlled reliable, complementary, adversarial, harmful, and
  unstable-relation scenarios using an untouched test split.
- `cmpb_conservative_gate_sensitivity.py` varies the minimum validation
  C-index gain required for graph eligibility in the right-censored stress
  test.
- `cmpb_repeated_train_only_graph_audit.py` rebuilds every graph inside one
  prespecified training split for one cancer and records fixed-versus-
  reconstructed contrasts at three gate margins.
- `assemble_repeated_train_only_graph_audit.py` combines completed
  cancer-by-split audits and computes cancer-clustered intervals.
- `cmpb_full_stability_baseline_figure.py` regenerates the complete
  six-method cancer-level stability comparison and its exact paired
  MKG--Uni-Cox cancer-cluster bootstrap intervals.
- `cmpb_merge_and_plot.py` merges the completed five-arm workers, recomputes
  cancer-bootstrap summaries, and renders the submission figure with every
  cancer displayed directly.
- `make_cmpb_graphical_abstract.py` regenerates the simplified four-stage
  vector graphical abstract and its PDF, SVG, PNG, and TIFF exports.

The sensitivity, repeated-audit, complete-stability, and five-arm scripts
generate their own source tables or figures. Numerical source tables are
supplied for historical plots whose original figure-generation scripts
remain outside the public release.
