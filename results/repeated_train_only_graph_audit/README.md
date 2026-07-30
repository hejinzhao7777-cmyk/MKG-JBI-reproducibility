# Repeated train-only graph-reconstruction audit

This directory contains the strict 6-cancer by 3-split audit reported in the
CMPB submission. Every run reconstructs expression scaling and all three
relation graphs within the training partition before routing, Top-20
selection, reduced ridge-Cox fitting, and held-out evaluation.

Primary zero-margin result:

- 18 prespecified cancer-by-split audits;
- mean fixed-minus-reconstructed held-out C-index: -0.001505;
- cancer-clustered 95% interval: [-0.009619, 0.005814];
- mean Top-20 Jaccard: 0.841259;
- route agreement: 13/18.

`runs/` contains every JSON and row-level CSV. The top-level raw,
cancer-aggregated, and overall summary tables are generated with:

```bash
python code/assemble_repeated_train_only_graph_audit.py \
  --root results/repeated_train_only_graph_audit/runs \
  --outdir results/repeated_train_only_graph_audit
```

The assembler refuses incomplete, duplicated, or configuration-mismatched
inputs and records SHA-256 hashes for all 18 JSON files. The audit bounds
graph-construction sensitivity; it does not prove the fixed full-cohort
graphs leakage-free or replace repeated nested cross-validation.
