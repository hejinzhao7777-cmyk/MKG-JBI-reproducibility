# Complete six-method stability comparison

This directory contains the submission-facing summary of within-cohort
Top-20 selection stability across six TCGA cancers.

- `Fig4_stability.pdf`: vector main-text figure; every cancer-level value is
  shown directly and the black diamond is the six-cancer mean.
- `CMPB_STABILITY_SIX_METHOD_SUMMARY.csv`: mean, median, and between-cancer
  SD for normalized RBO@20 and Jaccard.
- `CMPB_STABILITY_MKG_VS_UNICOX.json`: paired MKG--Uni-Cox differences,
  exact two-sided Wilcoxon tests, and exact percentile intervals over all
  \(6^6\) cancer-cluster bootstrap resamples.

The underlying values are in
`../source_tables/TableS_expanded_stability_baselines_jbi.csv`. The `jbi`
suffix is a historical provenance identifier and does not change the
analysis or target-journal package.
