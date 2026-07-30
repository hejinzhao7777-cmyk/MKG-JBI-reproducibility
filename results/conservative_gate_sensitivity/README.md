# Conservative gate sensitivity

This directory contains the right-censored minimum-gain sensitivity audit.

- `CMPB_CONSERVATIVE_GATE_RAW.csv`: one row per scenario, repetition, and gate
  margin.
- `CMPB_CONSERVATIVE_GATE_SUMMARY.csv`: Monte Carlo means, uncertainty
  intervals, support recovery, and no-relation frequencies.
- `CMPB_CONSERVATIVE_GATE_PAIRED.csv`: paired contrasts against the locked
  zero-margin gate.
- `CMPB_CONSERVATIVE_GATE_SUMMARY.json`: configuration and machine-readable
  summary.

The result is deliberately reported as a limitation. Increasing a fixed gain
margin raised abstention but did not improve the all-harmful test result.
