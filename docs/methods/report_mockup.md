# GAUGE paper-bundle mockup

This is the interpretation contract for the paper-facing bundle. Values below
are fictional layout examples, not TC results.

> `WITHIN_REFERENCE_RANGE` means only that the discrepancy did not exceed the
> declared empirical real-reference quantile. It is not proof of equality,
> physical realism, or application acceptability.

| Question | Metric / condition | Raw discrepancy (central comparison range) | Real-reference q50 → q95 | Percentile | Gap | Reference status | Control status |
|---|---|---:|---:|---:|---:|---|---|
| Physical fidelity | W1 core temperature / C2 | 0.31 K [0.22, 0.40] | 0.18 → 0.29 K | 97.4% | 1.18 | ABOVE_REFERENCE_RANGE | CONTROL_VALIDATED |
| Spatial structure | radial log-PSD / C2 | 0.84 [0.61, 1.11] dex² | 0.09 → 0.15 dex² | 100% | 12.5 | ABOVE_REFERENCE_RANGE | CONTROL_VALIDATED |
| Conditional fidelity | W1 cold area / C5 | — | — | — | — | REFERENCE_UNRESOLVED | CONTROL_VALIDATED |
| Diversity | coverage deficit / C2 | 0.07 [0.04, 0.10] | 0.05 → 0.08 | 81.0% | 0.67 | WITHIN_REFERENCE_RANGE | RESPONSIVE_BELOW_REFERENCE_RANGE |

Every card also prints:

- block unit, estimand, evaluation sample count, block count, repetition count,
  seed, and config hash;
- exact functional definitions and valid domains;
- conditional Monte Carlo intervals for the reference quantiles and gap;
- warnings for thin, unstable, invalid, or control-failed results;
- registered sensitivity/invariance curves and boundary-crossing severity;
- links to the schema-v3 report and versioned analysis protocol.

The empirical-reference gap profile is the primary summary. Its axis explicitly
states that it is a descriptive finite-archive coordinate, every point links to
a native-unit result, and no average score or global pass/fail count is produced.
