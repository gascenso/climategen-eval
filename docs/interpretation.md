# Interpreting GAUGE output

Read each GAUGE result in this order:

1. **Estimand and sample size.** A block-balanced result represents a uniformly
   selected dependence block followed by a uniformly selected frame. A
   frame-weighted result represents a uniformly selected frame. Detection
   resolution changes with sample and block counts.
2. **Raw discrepancy and uncertainty.** Functional W1 remains in the
   functional’s native units. This is the physically interpretable evidence;
   feature and curve metrics retain their registered units and definitions.
3. **Real-reference gap.** Gap 0 is the median real-vs-real discrepancy and gap
   1 is the pointwise reference boundary at the declared `N`.
4. **Detection.** `DEPARTURE_DETECTED` says the model statistic exceeded that
   finite-sample boundary. `NO_DEPARTURE_DETECTED` is a failure to detect, not
   evidence of equality or physical realism.
5. **Controls and provenance.** A metric that misses its registered positive
   control is not interpreted, even if the model appears favorable.

For conditional generators, inspect the condition-mixture audit before the
global row. If real and generated label proportions differ, the global result
is a composition check; use the matched conditional rows for model fidelity.

The reference-calibrated gap is:

```text
gap = (model discrepancy - real-reference median) /
      (pointwise 95% reference boundary - real-reference median)
```

It is a detectability coordinate, not a physical-severity scale. Do not compare
it across sample sizes, average it into a score, or use reference-envelope
overlap as an equivalence claim. Corrupted or degenerate reference lanes on the
same scale are contextual anchors, not universal acceptability thresholds.

## Conditional display

If `K <= 8`, the paper profile shows every native class and every predeclared
priority aggregate. For larger `K`, it shows the native-class median/range,
labels the largest observed gap, retains priority strata, and exports the exact
`K`-column matrix to the appendix. “Largest observed” is descriptive unless a
worst-class claim was preregistered. Pooled strata never replace their component
rows.

In the copying result, a positive generated-minus-real nearest-to-training
effect means generated queries are farther from training than held-out real
queries. It does not by itself prove physical inadmissibility. When labels
exist, the test must match them within each feature-region comparison.
