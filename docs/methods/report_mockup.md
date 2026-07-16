# GAUGE paper-bundle mockup

This is the interpretation contract for the generated paper-facing bundle. The
final layout may change, but the information hierarchy and wording may not.

All values below are fictional layout examples, not TC results and not candidate
tolerances. They must not be used to choose case-study thresholds.

> Read the status columns separately. “No departure detected” means that this
> evaluation did not detect a discrepancy at the printed sample size. It is not
> proof of equality. “Within tolerance” is issued only from a direct confidence
> bound relative to the declared application tolerance.

| Question | Metric / stratum | Raw effect (95% interval) | Real-reference envelope | Tolerance | Detection | Practical relevance |
|---|---|---:|---:|---:|---|---|
| Physical fidelity | W1 core temperature / global | 0.31 K [0.22, 0.40] | q50 0.18; q95 0.29 K | 0.50 K | DEPARTURE_DETECTED | WITHIN_TOLERANCE |
| Physical fidelity | radial log-PSD / global | 0.84 log-PSD² [0.61, 1.11] | q50 0.09; q95 0.15 | — | DEPARTURE_DETECTED | NOT_APPLICABLE |
| Conditional fidelity | W1 cold area / Cat 5 | — | — | 1.0 percentage point | NOT_EVALUATED | UNDERPOWERED |
| Diversity | coverage deficit / global | 0.07 [0.04, 0.10] | q50 0.05; q95 0.08 | — | NO_DEPARTURE_DETECTED | NOT_APPLICABLE |
| Memorization | localized NN / region 3 | −0.14 [−0.21, −0.07] | held-out real to training | — | COPYING_DETECTED | NOT_APPLICABLE |

Every card also prints:

- the block unit, estimand, `N_eval`, test-block count, resample count, seed, and config hash;
- the exact functional definition and support/tolerance justification;
- warnings where a result is thin, unvalidated, or based on a diagnostic interval;
- the sensitivity perturbation and severity response for each intended metric;
- links to the versioned JSON report and frozen preregistration.

The reference-calibrated gap profile is the primary paper-facing summary. It
labels its x-axis “detectability, not physical severity,” links each point to
its raw native-unit result, and never presents an average score. The complete
native-unit card remains part of the bundle for audit and interpretation.
