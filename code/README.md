# Analysis scripts

The scripts in this folder are the source files used for the locked analyses. `final_config_comparison.py` is configured for portable paths:

```bash
set MKG_DATA_ROOT=D:\\path\\to\\processed_data
set MKG_OUTPUT_ROOT=D:\\path\\to\\mkg_outputs
python final_config_comparison.py LUAD
```

The remaining scripts preserve the locked submission workflow and its specific audit calculations. Some audit scripts were executed from the original project layout; before an end-to-end rerun, review their path constants and point them to the same `MKG_DATA_ROOT` and `MKG_OUTPUT_ROOT` locations. This explicit note is intentional: it prevents a misleading claim that the public release is turnkey before the full preprocessing pipeline and distributable processed matrices have been deposited.

`results/` contains the locked outputs used by the manuscript, so numerical claims can be inspected without rerunning the compute-intensive workflow.

