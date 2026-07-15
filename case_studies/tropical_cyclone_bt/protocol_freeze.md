# TC scientific protocol freeze

Frozen on 2026-07-15 before running the final GAUGE analysis or any restricted
Track-2 generator fit. This document freezes scientific choices; it does not
assert that required artifacts or positive controls already exist. Any change
after final outcomes are inspected must be logged as an amendment and the
affected result labeled exploratory.

## Data and estimand

- All intensity labels and continuous targets are IBTrACS `USA_WIND` one-minute
  sustained winds. Mixed-period `WMO_WIND` is prohibited.
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
must report `UNDERPOWERED` when its shift resolution is coarser than a declared
tolerance and must not reinterpret non-detection as equivalence.

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

- Execute and pass the frozen Track-1 sensitivity and copyist controls.
- Produce three restricted generator fits and their checksum-bound provenance.
- Archive permission-cleared real/generated artifacts.
- Preserve this record in Git history and record any amendments.

The literature survey is a separate manuscript workstream and is not part of
this scientific/data-processing freeze.
