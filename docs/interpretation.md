# Interpreting a GAUGE card

Read each cell in this order:

1. **Estimand and sample size.** A block-balanced result represents a uniformly
   selected dependence block followed by a uniformly selected frame. A
   frame-weighted result represents a uniformly selected frame. Results at
   different sample sizes have different detection resolution.
2. **Raw effect and interval.** For functional W1 these remain in the functional's
   native units. The upper bound, not the reference-position score, determines
   `WITHIN_TOLERANCE`.
3. **Detection.** `DEPARTURE_DETECTED` says the model statistic exceeded the
   finite-N perfect-generator reference boundary. `NO_DEPARTURE_DETECTED` is a
   failure to detect, not evidence of equality.
4. **Practical relevance.** `WITHIN_TOLERANCE` and `EXCEEDS_TOLERANCE` require
   the entire direct confidence interval to lie on one side of the predeclared
   tolerance. Crossing the tolerance is `INCONCLUSIVE`; inadequate declared
   shift resolution is `UNDERPOWERED`.
5. **Controls and provenance.** A metric that misses its registered positive
   control is not interpreted, even if the model appears favorable.

Canonical combinations:

| Detection | Practical | Permitted wording |
|---|---|---|
| no departure | within tolerance | “No departure was detected at N, and the upper bound was below the declared tolerance.” |
| departure | within tolerance | “A statistically detectable but practically tolerated departure was found.” |
| departure | exceeds tolerance | “The departure was detected and exceeded the declared tolerance.” |
| either | inconclusive | “The data did not resolve which side of the tolerance contains the discrepancy.” |
| no departure | underpowered | “No departure was detected, but the design was insufficiently sensitive at the tolerance.” |

Never write “indistinguishable,” “equivalent,” or “physically realistic” from a
reference-envelope overlap alone. The standardized reference-position plot is a
secondary detectability diagnostic and cannot compare physical severity across
metrics or sample sizes.
