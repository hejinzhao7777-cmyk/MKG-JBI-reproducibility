# Independent multi-omics survival-cohort screen

Screen date: 2026-07-31

## Decision rule

A technically eligible cohort had to provide, for the same patients:

1. gene-level mRNA expression;
2. gene-level DNA methylation;
3. gene-level copy-number alteration;
4. overall-survival time and status;
5. at least 80 complete cases and 20 deaths; and
6. public participant-level access sufficient for reproducible analysis.

For inclusion as a formal end-to-end held-out audit, the cohort also had to
leave enough events in each untouched test partition for a 20-feature survival
score. This second requirement concerns inferential stability rather than mere
file availability.

## Sources searched

- cBioPortal public study catalogue and REST API:
  <https://www.cbioportal.org/datasets>
- cBioPortal Datahub:
  <https://github.com/cBioPortal/datahub>
- NCI Genomic Data Commons cases API:
  <https://api.gdc.cancer.gov/cases>
- CPTAC collections in The Cancer Imaging Archive:
  <https://www.cancerimagingarchive.net/>

The search prioritized independent cohorts corresponding to the six locked
TCGA cancers, then expanded to other public cancer cohorts when no large
same-cancer three-omics survival stack was available.

## Audited candidates

| Study | Complete three-omics survival set | Deaths | Decision | Reason |
|---|---:|---:|---|---|
| METABRIC breast cancer | 1,416 | 830 | Include | Large independent cohort; RNA, promoter RRBS methylation, discrete CNA, and complete OS support five train-only held-out splits. |
| CPTAC lung adenocarcinoma | 93 | 23 | Do not use as formal audit | Same-cancer and technically eligible, but a 25% test split contains only about six deaths; it is unlikely to reduce inferential risk. |
| CPTAC colon cancer | 0 with all three layers | - | Exclude | RNA, CNA, and OS were present, but the public study profile lacked methylation. |
| CPTAC renal cell carcinoma | 0 with all three layers | - | Exclude | RNA, CNA, and OS were present, but the public study profile lacked methylation. |
| OncoSG LUAD/STAD | 0 with all three layers | - | Exclude | Public profiles did not provide the complete RNA-methylation-CNA-OS stack. |
| Other public LIHC/HNSC cohorts inspected in cBioPortal | 0 with all three layers | - | Exclude | At least one required molecular layer or usable survival outcome was absent. |

## METABRIC analysis lock

- The original six-cancer analyses and their reported averages remain
  unchanged.
- The METABRIC audit is secondary and is not used to retune the locked
  hyperparameters.
- The feature universe is the intersection of the six locked TCGA expression
  universes and genes available in all three METABRIC assays.
- No outcome-based filtering is used to define the feature universe.
- For each prespecified split, imputation, scaling, all graph construction,
  routing, Top-20 selection, and reduced Cox fitting use training data only.
- The untouched test partition is used once for the paired MKG versus
  zero-graph C-index comparison.
