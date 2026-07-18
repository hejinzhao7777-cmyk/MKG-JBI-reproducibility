# Public data sources

## Training cohorts

Training data were obtained from the public TCGA program through the NCI Genomic Data Commons (GDC). The cancer types are LUAD, LIHC, KIRC, COAD, STAD, and HNSC. Molecular inputs include RNA-seq expression, DNA methylation, copy-number information, and survival annotations as described in the manuscript.

- GDC Data Portal: <https://portal.gdc.cancer.gov/>
- TCGA program information: <https://www.cancer.gov/ccg/research/genome-sequencing/tcga>

## Independent external validation cohorts

| Cancer type | Primary validation cohort | Repository |
| --- | --- | --- |
| LUAD | GSE31210 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE31210 |
| LIHC | GSE14520 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14520 |
| KIRC | GSE29609 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE29609 |
| COAD | GSE39582 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39582 |
| STAD | GSE84437 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84437 |
| HNSC | GSE65858 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65858 |

An additional LUAD cohort (GSE50081), LIHC cohort (GSE76427), and KIRC cohort (E-MTAB-1980) were used where described in the analytical code and manuscript.

## Local input layout

Do not upload the downloaded or processed patient-level matrices to this GitHub repository. Set `MKG_DATA_ROOT` to a local directory containing the study's processed input layout. The original analysis used cancer-specific folders under `processed_data/`. A DOI-minting data archive should contain the exact processed matrices and preprocessing manifest after the authors have verified that redistribution complies with each source repository's terms.

