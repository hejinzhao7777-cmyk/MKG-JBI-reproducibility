# PGD iteration provenance correction

The first strict graph-reconstruction audit recorded a requested PGD cap of
150. In the legacy core function, that default was bound when the module was
imported at 300 iterations. Changing the module variable afterward did not
change the bound default, so the effective cap for both stage-1 and stability
fits was 300.

The numerical results are unchanged. The JSON metadata now records both the
requested and effective values. The public audit script propagates the
iteration setting before importing the core module, and the repeated audit
passes the value explicitly.
