# TC scientific protocol freeze

Frozen on 2026-07-15 before running the final GAUGE analysis. This document
freezes scientific choices; it does not
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
- The evaluation targets the block-balanced distribution: sample a storm uniformly,
  then a frame uniformly within that storm. Every registered stratum uses all
  available generated rows (6,000 globally; 1,000 per native class).
- Native TS/C1--C5 strata are reported. `C4_plus=[C4,C5]` was additionally
  registered to stabilize an extreme-tail descriptive row when the C5 test
  reference is thin (85 frames, 22 storms). It does not replace the separate
  C4/C5 rows and is not the standard major-hurricane grouping; that terminology
  was corrected in the amendment below.

## Registered metrics and controls

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
each image's discrete Fourier power to the declared numerical threshold. The
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

## Reference validation and final-run amendments

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

The generic Monte Carlo gate validates whole-block partition invariants,
pointwise false-positive behavior, simultaneous boundaries, unequal block
sizes, and controlled-shift power. The TC training-only response and copyist
controls pass their registered contracts. Shift resolution remains a design
sensitivity result, never an equivalence verdict.

Framework-scope amendment on 2026-07-21, after the final run: the separate
application-materiality verdict branch was retired from GAUGE v1. Requiring
user-chosen thresholds conflicts with the one-call adoption goal; inferring
them from the real-reference boundary or corruption lanes would be circular or
arbitrary. This changes no raw discrepancy, reference envelope, gap,
sensitivity control, memorization result, or detection conclusion. The earlier
implementation remains transparent in Git history.

## Gates still blocking final inference

- Independently review the real-reference engine, simultaneous inference, and
  interpretation wording; the frozen training-only sensitivity and copyist
  controls have passed.
- Archive permission-cleared real/generated artifacts.
- Preserve this record in Git history and record any amendments.

The literature survey is a separate manuscript workstream and is not part of
this scientific/data-processing freeze.
