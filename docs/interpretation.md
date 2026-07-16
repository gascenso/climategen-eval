# Interpreting a GAUGE card

Read each cell in this order:

1. **Estimand and sample size.** A block-balanced result represents a uniformly
   selected dependence block followed by a uniformly selected frame. A
   frame-weighted result represents a uniformly selected frame. Results at
   different sample sizes have different detection resolution.
   For conditional generators, also inspect the condition-mixture audit. If label
   proportions differ, the global row is a composition check; read the matched
   conditional rows for model fidelity.
2. **Raw effect and interval.** For functional W1 these remain in the functional's
   native units. The upper bound, not the reference-position score, determines
   `WITHIN_TOLERANCE`.
3. **Detection.** `DEPARTURE_DETECTED` says the model statistic exceeded the
   finite-N perfect-generator reference boundary. `NO_DEPARTURE_DETECTED` is a
   failure to detect, not evidence of equality.
4. **Practical relevance.** `WITHIN_TOLERANCE` and `EXCEEDS_TOLERANCE` require
   the entire direct confidence interval to lie on one side of the predeclared
   tolerance. Crossing the tolerance is `INCONCLUSIVE`; it is `UNDERPOWERED`
   when the uncertainty margin alone is at least as large as the tolerance or
   the stratum lacks the minimum number of blocks. Detection sensitivity is
   reported separately and does not change this verdict.
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
reference-envelope overlap alone. The reference-calibrated gap can be the
primary *presentation* coordinate while remaining a detectability diagnostic:

```text
gap = (model discrepancy - real-reference median) /
      (pointwise 95% reference boundary - real-reference median)
```

Thus gap 0 is the median real-vs-real discrepancy and gap 1 is the pointwise
boundary. It cannot compare physical severity across metrics or sample sizes,
must not be averaged into a model score, and never replaces the native-unit
interval for practical conclusions.

For conditional generators, the default paper profile never substitutes an
unmatched global mixture for class fidelity. If `K <= 8`, show every native
class and every preregistered priority aggregate. For larger `K`, show the
native-class median/range, label the largest observed gap, retain all priority
strata, and put the exact `K`-column matrix in the appendix. “Largest observed”
is descriptive unless worst-class inference was preregistered. The complete
native-unit table must remain available regardless of how the main figure is
compressed.

In the copying result, a positive generated-minus-real nearest-to-training
effect means generated queries are farther from training than held-out real
queries. The v0.1 machine label `OFF_MANIFOLD` is retained for schema
compatibility, but does not by itself prove physical inadmissibility. When query
labels exist, the test must match them within the feature-region comparison.
