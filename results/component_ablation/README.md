# Five-arm component ablation

These files support the five-arm no-graph, equal-weight, stability-only,
utility-only, and joint-MKG comparison.

- `CMPB_FIVE_ARM_ABLATION.csv`: weights and frozen-score C-indices for all six cancers.
- `CMPB_FIVE_ARM_STABILITY.csv`: RBO@20 and Jaccard from 30 bootstrap lists.
- `CMPB_FIVE_ARM_STATISTICAL_SUMMARY.json`: 50,000-resample cancer-bootstrap
  summaries and paired contrasts.
- `CMPB_COMPUTATIONAL_COST.csv`: measured fit time and peak process RSS for
  newly fitted signatures.
- `CMPB_SYNTHETIC_FIVE_ARM_RAW.csv`: 50-repetition independent-test results.
- `CMPB_SYNTHETIC_FIVE_ARM_SUMMARY.csv`: scenario and scheme means.
- `CMPB_SYNTHETIC_ROUTING_WEIGHTS_RAW.csv`: routing decisions in every
  controlled repetition.

The primary interpretation is deliberately bounded. Positive out-of-fold
utility supplies the main rejection safeguard. Stability modifies allocation
when multiple graph layers remain eligible; it does not show a systematic
external C-index advantage over utility-only routing in these data.
