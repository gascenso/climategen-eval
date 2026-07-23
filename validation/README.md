# Statistical validation artifacts

Regenerate the current gate and its paper-ready PDF/PNG figure with:

```bash
MPLCONFIGDIR=/tmp/mpl-gauge python scripts/validate_gauge_statistics.py \
  --trials 300 --resamples 200 --jobs 4
```

The gate uses Wilson Monte Carlo uncertainty and validates the scientific claims
GAUGE actually exposes:

- whole-block reference-plan invariants;
- numerical parity between the vectorized validation calculation and the public
  reference engine;
- conservative pointwise false-positive behavior for balanced, severely
  unequal, C5-like 22-block, and minimum-valid 20-block designs;
- location-shift power curves in every block regime;
- response to variance narrowing, excess variance, crossing CDFs, tail
  contamination, and a train/test population shift;
- at least 80% power for a declared controlled location shift;
- familywise behavior and finite Bonferroni order-statistic boundaries for a
  registered family; and
- fail-closed behavior below the 20-block minimum.

The real-reference detector is expected to be conservative because both sides
of its reference comparison inherit finite-block uncertainty. The observed rate
is reported rather than described as exactly calibrated to `alpha`.

The C5-like regime deliberately records low power despite `N=1,000`: with only
22 independent storms, drawing more frames cannot replace missing block-level
information. This is a design-sensitivity result, not a defect hidden by the
software.

Joint claims require at least `family_size / alpha - 1` reference resamples.
GAUGE withholds the simultaneous result when the empirical tail cannot resolve
that boundary.
