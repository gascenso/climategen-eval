# Statistical validation artifacts

Regenerate the v0.1 gate with:

```bash
PYTHONPATH=src python scripts/validate_gauge_statistics.py \
  --trials 500 --resamples 200 \
  --output validation/statistical_validation_v0.1.json
```

The gate uses Wilson Monte Carlo uncertainty and validates the scientific claims
GAUGE actually exposes:

- whole-block reference-plan invariants;
- conservative pointwise false-positive behavior under equality;
- equality behavior with severely unequal block sizes;
- at least 80% power for a declared controlled location shift;
- response to variance change and tail contamination; and
- finite Bonferroni order-statistic boundaries for a registered family.

The real-reference detector is expected to be conservative because both sides
of its reference comparison inherit finite-block uncertainty. The observed rate
is reported rather than described as exactly calibrated to `alpha`.

Joint claims require at least `family_size / alpha - 1` reference resamples.
GAUGE withholds the simultaneous result when the empirical tail cannot resolve
that boundary.
