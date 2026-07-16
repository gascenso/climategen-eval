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

Power is evaluated at every exact preregistered tolerance rather than inferred
from the nearest generic sweep point. Twelve of the 56 registered
stratum-functional cells attain at least 80% shift-detection power there. They
are global axisymmetry, DAV, eye contrast, and radius of maximum gradient; TS
axisymmetry, DAV, and eye contrast; C1 axisymmetry and DAV; C2 axisymmetry; C4
axisymmetry; and C4+ axisymmetry. The other 44 cells lack that declared
detection sensitivity. This is a property of the evaluation design and
perturbation lane, not the diffusion model and not an equivalence result.

For regular bounded functionals, the validated whole-block influence interval
places the training-only pseudo-generator within tolerance in 16 of 56 cells:
all eight axisymmetry cells, five DAV cells, two eye-contrast cells, and global
minimum-core BT. The discrete maximum-gradient-radius functional retains the
finite-sample DKW fallback. The remaining cells legitimately retain
`INCONCLUSIVE` or `UNDERPOWERED` as possible final outcomes; no tolerance was
relaxed in response to this artifact.

These 56 cells must not be cross-tabulated with the legacy tc-diffusion
"AT FLOOR" total. The legacy total was a model-versus-real finite-N detection
screen over a different set of metrics and strata; this artifact is a
training-only power and reference-mechanics calibration with no model samples.
