# TC scientific protocol freeze

Frozen on 2026-07-15 before running the final GAUGE analysis or any restricted
Track-2 generator fit. This document freezes scientific choices; it does not
assert that required artifacts or positive controls already exist. Any change
after final outcomes are inspected must be logged as an amendment and the
affected result labeled exploratory.

## Data and estimand

- All intensity labels and continuous targets are IBTrACS `USA_WIND` one-minute
  sustained winds. Mixed-period `WMO_WIND` is prohibited.
- Raw Kelvin archives remain checksum-preserved, but every GAUGE analysis view is
  clipped lazily to 170--325 K before metrics or train-fitted features. These are
  the generator-training 0.05th/99.95th-percentile preprocessing limits, not
  claims about the physically possible BT range.
- Train, validation, and test partitions are storm-disjoint. The existing
  corrected test split is retained unchanged.
- Track 1 targets the block-balanced distribution: sample a storm uniformly,
  then a frame uniformly within that storm. Every registered stratum uses all
  available generated rows (6,000 globally; 1,000 per native class).
- Native TS/C1--C5 strata are reported. `C4_plus=[C4,C5]` is additionally
  registered because both are major hurricanes and the C5 test reference is
  thin (85 frames, 22 storms). It does not replace the separate C4/C5 rows.

## Track-1 practical tolerances

The tolerances are protocol materiality/resolution thresholds, not estimates
from the held-out outcomes: 1 K for BT-like functionals and RMS asymmetry;
1 percentage point for cold-cloud fraction; one 500/32 = 15.625 km radial bin
for maximum-gradient radius; and 100 deg² for deviation-angle variance. GAUGE
reports `UNDERPOWERED` when confidence uncertainty alone is at least as large
as the declared tolerance or a stratum fails the minimum-block rule. Shift
resolution is reported separately and never substitutes for direct inference;
non-detection is not reinterpreted as equivalence.

The joint primary detection family contains the seven scalar physical
functionals across global, six native-class, and C4+ strata (56 cells). Curve,
feature-space, dependence, admissibility, and memorization results remain
registered diagnostics rather than members of this primary simultaneous family.

The train-fit feature space uses seven standardized scalar functionals plus 20
EOF coefficients fitted to a deterministic 512-frame training subset. MMD
bandwidth and all normalization are training-only. The localized copying test
uses eight train-fit regions, whole-test-storm resampling, and the exact-copyist
positive control.

Sensitivity sweeps are frozen in `preregistration.yaml`. Failure of an expected
metric response withholds interpretation for that metric; it does not authorize
changing the perturbation after seeing final model results.

The controls use 80 deterministically selected training storms and a
block-balanced sample size of 1,000, matching a native-class evaluation cell.
Real-reference boundaries use 500 whole-block resamples at the 95th percentile.
Warm shift, Gaussian blur, and center jitter are paired severity sweeps. Full
phase shuffling must both trigger the registered structure metrics and preserve
each image's discrete Fourier power to the declared numerical tolerance. The
localized copying procedure must detect an exact-copyist control using the same
512-frame feature-fit limit and eight regions as the registered analysis.

For severity sweeps, response validity and resolution are separate: a metric
passes its response contract when the Spearman severity trend is at least 0.8.
Whether the maximum frozen severity exceeds the real-reference boundary is
reported separately and may reveal limited sensitivity at this N. Binary phase
and copyist positive controls still require boundary crossing or copying
detection, respectively.

Amendment, before any final test or generated outcome was inspected: naïve
geometric PSD edges produced empty low-frequency bins on the 256×256 discrete
Fourier grid. The frozen implementation therefore selects 24 distinct observed
radial wavenumbers nearest the log-spaced targets and uses geometric-midpoint
boundaries. This retains 24 ordered bins in physical cycles/km while guaranteeing
that every bin is defined.

Amendment after a training-only calibration failure, before final test or
generated outcomes: the raw archive spans approximately 117--348 K, whereas the
generator training pipeline clips real fields to 170--325 K before normalization.
GAUGE had incorrectly treated those preprocessing quantiles as raw support while
feeding unclipped real fields to metrics and EOFs. All analysis inputs now use a
lazy 170--325 K clipped view, exactly matching the learned field representation;
raw files and checksums remain unchanged. Boundary saturation is reported as a
diagnostic rather than calling raw values outside the preprocessing quantiles
physically inadmissible.

## Track-1 calibration result

The frozen training-only calibration uses 2,000 primary reference resamples per
stratum, 200 reference draws per shift-resolution curve, 80% target power, and the exact
registered evaluation sizes. It reads no test fields or functional values.
Published test-stratum block counts are used only to choose conservative numbers
of training storms. Every pseudo-generator pool is count-matched and storm-
disjoint from its real-reference pool.

The final runner retains all 2,000 draws for functional W1 and the preregistered
simultaneous family. Secondary curve, feature-space, dependence, and admissibility
diagnostics use 200 draws; quadratic feature-set metrics use deterministic matched
subsamples of 256 fields per side per draw. Each cell reports these effective
counts. Complete partition plans are deterministic and recorded by seed and
SHA-256 with a three-draw preview, avoiding an unusably large JSON expansion.
This computational amendment followed an aborted invocation that produced no
metric or report output and changes no primary estimand or decision rule.

The first completed final-output audit exposed a separate target-mixture issue
already visible in the frozen manifests: generated data contain exactly 1,000
fields per class, while the real test archive follows its natural, strongly
imbalanced intensity distribution. All global rows are retained as supplied-bank
composition checks, but conditional-model fidelity is concluded only from the
native-class and predeclared C4+ rows. The same mixture cannot be allowed to
confound memorization, so the superseding final run tests copying within joint
intensity-class-by-feature-region cells and controls multiplicity across all
supported cells. The original completed report and its SHA-256 are retained;
no physical metric or conclusion rule was changed after seeing its values.

All reference-mechanics gates pass, including whole-block partition invariants,
reference-boundary stability, finite pointwise and simultaneous thresholds, and
zero detections for the registered pseudo-generator. Power is evaluated at the
exact tolerance in every cell: 12 of 56 attain at least 80% shift-detection
power and 44 do not. This is training-only design sensitivity, not a model
score. The original all-DKW practical interval was valid but vacuous for this
case study. Before accessing generated test outcomes, it was replaced for
regular bounded scalar functionals by the validated whole-block influence
interval described above; the discrete maximum-gradient radius retains DKW.
Tolerances, models, strata, sample sizes, and detection rules were unchanged.

## Track 2

Three whole-storm holdouts were selected uniformly with seed 1789 from 97 storms
meeting outcome-blind eligibility rules: at least 20 frames, at least 10 C4+
frames, at least one C5 frame, and every exact Vmax value available elsewhere in
the restricted training partition. The selected storms are:

- `2006259N19155`
- `2016273N13300`
- `2018269N11220`

Each holdout requires a fresh `baseline_ztsnr` generator fit with every frame of
the storm removed. Generated conditions must be exact Vmax values from the
restricted training support. Copy, generated, and restored-real additions are
matched by exact Vmax and count; category-only matching is prohibited because it
would confound the primary regression comparison.

The primary endpoint is storm-balanced Vmax MAE with a 2.5 kt equivalence margin
(half the dominant 5 kt reporting increment). Native-category accuracy is
secondary with a two-percentage-point margin. The headline contrast is
generated+geometry versus copy+geometry; generated versus copy isolates source,
and restored-real+geometry versus copy+geometry measures empirical headroom.

The CNN architecture and training budget follow the existing `tc-diffusion`
evaluator experiment (27 maximum epochs, batch size 16, fixed restricted-real
samples per epoch). Checkpoint selection uses storm-balanced validation Vmax MAE
to align with the primary endpoint. Five paired seeds are fixed. Geometry is
restricted to right-angle rotations and reflect-padded translations of at most
two pixels, applied only to added rows.

## Gates still blocking final inference

- Independently review the completed TC-specific statistical calibration and
  resolve the direct-W1 power limitation before the final test run; the frozen
  training-only sensitivity and copyist controls have passed.
- Produce three restricted generator fits and their checksum-bound provenance.
- Archive permission-cleared real/generated artifacts.
- Preserve this record in Git history and record any amendments.

The literature survey is a separate manuscript workstream and is not part of
this scientific/data-processing freeze.
