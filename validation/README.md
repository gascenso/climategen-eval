# Statistical validation artifacts

Regenerate the v0.1 gate with:

```bash
PYTHONPATH=src python scripts/validate_gauge_statistics.py \
  --trials 200 --resamples 200 \
  --output validation/statistical_validation_v0.1.json
```

The gate uses Wilson Monte Carlo uncertainty. It requires the lower confidence
bound for DKW coverage and declared-shift power to meet their targets, and the
upper confidence bound for equality false positives to remain below alpha. The
real-reference detector is expected to be conservative because both sides of
its reference comparison inherit finite-block uncertainty; the observed rate is
reported rather than described as exactly calibrated to alpha.

Joint claims use Bonferroni cell-wise order-statistic boundaries. This avoids
assuming that conditional strata share a valid resampling coupling. Because a
finite Monte Carlo sample can only resolve tail probabilities of order
`1 / (K + 1)`, GAUGE withholds the simultaneous conclusion unless
`K >= family_size / alpha - 1`.

The classical clustered percentile bootstrap is included as a diagnostic and
cannot issue v0.1 tolerance conclusions. Thin-stratum behavior and status
semantics are covered by unit tests; full TC-specific calibration remains a
case-study preregistration gate.
