# Frozen Track-1 statistical calibration

`track1_statistics.json` and `track1_statistics.pdf` are generated from the
corrected training archive with:

```bash
PYTHONPATH=src python case_studies/tropical_cyclone_bt/run_track1_calibration.py
```

The run uses the frozen protocol in `../calibration_protocol.yaml`. It reads no
test fields or test functionals. Previously published test-stratum block counts
are used only to select conservative numbers of training storms for the
calibration strata.
For each stratum, the pseudo-generator is sampled from a count-matched set of
training storms that is disjoint from the real-reference pool.

The reference-engine gates pass: block partitions are disjoint, reference
boundaries and simultaneous thresholds are finite and stable, the registered
pseudo-generator causes no pointwise or simultaneous detections, and all shift
resolution calculations are finite. This validates the mechanics of the frozen
reference experiment; it does not establish practical equivalence power.

Only 6 of the 56 registered stratum-functional cells have 80% shift-detection
resolution at or below the preregistered practical tolerance. They are global
DAV; TS DAV and eye contrast; C1 axisymmetry and DAV; and C2 axisymmetry. The
other 50 cells cannot establish within-tolerance performance at their declared
sample sizes with the current inference. The final analysis must preserve
`UNDERPOWERED` as a possible conclusion and must not describe a non-detection in
those cells as practical equivalence.

The distribution-free DKW bounds are retained as conservative direct W1 bounds.
An independent statistical review must decide whether GAUGE should additionally
register and validate a more powerful dependence-aware direct interval before
the final test run. Tolerances cannot be relaxed in response to this artifact.
