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

## Independent complete-stack audit

METABRIC breast cancer molecular and clinical profiles were obtained from the
public cBioPortal Datahub study `brca_metabric`. The audit used the Illumina
HT-12 v3 log-intensity expression profile, promoter RRBS beta values, discrete
copy-number calls, and overall survival.

- Study files: <https://github.com/cBioPortal/datahub/tree/master/public/brca_metabric>
- Original cohort: Curtis et al., Nature 2012, DOI
  <https://doi.org/10.1038/nature10983>
- Molecular landscape: Pereira et al., Nature Communications 2016, DOI
  <https://doi.org/10.1038/ncomms11479>
- Long-term follow-up: Rueda et al., Nature 2019, DOI
  <https://doi.org/10.1038/s41586-019-1007-8>
- Datahub database license: Open Data Commons Open Database License.

The source-file and processed-file SHA-256 values used by the audit are stored
in `results/metabric_multiomics_portability/METABRIC_PREPARATION_MANIFEST.json`.

## Local input layout

Do not upload downloaded or processed participant-level matrices to this
GitHub repository. Set `MKG_DATA_ROOT` to a local directory containing the
study's processed input layout. The original analysis used cancer-specific
folders under `processed_data/`. METABRIC preparation additionally accepts a
local raw-file cache and writes its own processed directory. Any future
participant-level archive requires a separate redistribution review under all
originating terms; the present repository distributes scripts, hashes, and
aggregate analysis outputs only.
