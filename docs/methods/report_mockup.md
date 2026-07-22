# GAUGE paper-bundle mockup

This is the interpretation contract for the paper-facing bundle. Values below
are fictional layout examples, not TC results.

> “No departure detected” means that this evaluation did not exceed the
> finite-`N` real-reference boundary. It is not proof of equality, physical
> realism, or universal application acceptability.

| Question | Metric / stratum | Raw discrepancy (95% resampling interval) | Real-reference median → boundary | Gap | Detection |
|---|---|---:|---:|---:|---|
| Physical fidelity | W1 core temperature / global | 0.31 K [0.22, 0.40] | 0.18 → 0.29 K | 1.18 | DEPARTURE_DETECTED |
| Physical fidelity | radial log-PSD / global | 0.84 [0.61, 1.11] log-PSD² | 0.09 → 0.15 | 12.5 | DEPARTURE_DETECTED |
| Conditional fidelity | W1 cold area / Cat 5 | — | — | — | NOT_EVALUATED |
| Diversity | coverage deficit / global | 0.07 [0.04, 0.10] | 0.05 → 0.08 | 0.67 | NO_DEPARTURE_DETECTED |

Every card also prints:

- block unit, estimand, `N_eval`, block count, resample count, seed, and config hash;
- exact functional definitions and valid domains;
- warnings for thin, unstable, or control-failed results;
- sensitivity perturbations and severity responses;
- links to the versioned JSON report and frozen preregistration.

The reference-calibrated gap profile is the primary summary. Its axis says
“detectability, not physical severity,” every point links to a raw native-unit
result, and no average score is produced.
