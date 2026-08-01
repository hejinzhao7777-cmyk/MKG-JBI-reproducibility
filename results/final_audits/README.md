# Final pre-submission audits

These files record the final checks added before the CMPB submission. They
contain aggregate or gene-level outputs only; participant-level matrices are
not redistributed.

- `outer_holdout_gate/`: routed-versus-zero comparisons on 18 untouched outer
  holdouts, aggregated cancer-first.
- `kirc_inclusive/`: primary five-cancer and KIRC-inclusive external summaries.
- `cv_tuned_cox/`: development-CV-selected Cox-Lasso and Cox elastic-net
  signatures, alpha traces, external scores, and patient-bootstrap intervals.
- `cv_tuned_cox_stability/`: conditional Top-20 stability of the two tuned Cox
  selectors on the shared subsamples.
- `additional_external_cohorts/`: complete reporting of GSE50081 and GSE76427.
- `pgd_convergence/` and `fista_component/`: numerical checks of the fixed
  300-update feature generator against a converged accelerated reference.
- `fista_routing/`: complete six-cancer high-accuracy routing reruns; all 150
  candidate and routed fits met the stated proximal-gradient tolerance.
- `directional_methylation_topology/`: six-cancer order-invariant
  directional-maximum graph construction diagnostics. Large adjacency files
  are deliberately excluded; the script and aggregate topology table are
  sufficient to reconstruct and audit them from the public inputs.
- `directional_routing/`: complete six-cancer downstream routing, Top-20, and
  primary external-C-index sensitivity to the directional-maximum graph.
- `conditional_stability/` and `cv_tuned_cox_stability/`: matched-subsample
  conditional fused-ranking stability results for all six methods.
- `updated_figure/`: final cross-audit summary tables and Figure 4 source data.

The submission lock remains the primary algorithmic configuration. These
folders expose sensitivity to cohort inclusion, graph construction, numerical
optimization, and outer graph admission rather than silently replacing the
lock.
