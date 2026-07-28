# Interpreting GAUGE output

Read each metric-by-condition result in this order:

1. **Population and weighting.** The default block-balanced result represents a
   uniformly selected event followed by a uniformly selected field from that
   event. A frame-weighted result instead represents a uniformly selected
   field. The report states which population was used.
2. **Native discrepancy.** For a scalar physical functional, W1 is expressed in
   that functional's units. Other metrics retain their registered definition
   and units.
3. **Central comparison range.** This is the central range of
   generated-versus-real discrepancies obtained as the admissible held-out-real
   subset changes. It is conditional on the supplied generated bank.
4. **Empirical real reference.** GAUGE repeatedly compares two event-disjoint
   real subsets using the same sample size, weighting, and condition. The report
   gives the median and 95th percentile of those discrepancies, together with
   the separately labelled upper boundary if the default 0.95 reference
   quantile was changed.
5. **Reference gap and percentile.** These say where the model discrepancy sits
   relative to the empirical real-reference distribution.
6. **Controls and warnings.** An affirmative interpretation is withheld when a
   metric fails its registered sensitivity or invariance control, or when the
   empirical reference is numerically unresolved.

The descriptive reference gap is

```text
gap = (model discrepancy - real-reference q50) /
      (real-reference q95 - real-reference q50)
```

Thus, under the default configuration, gap 0 marks the median real-versus-real
discrepancy and gap 1 marks its 95th percentile under the declared design. If
the upper quantile is changed, gap 1 marks that explicitly labelled percentile.
`WITHIN_REFERENCE_RANGE` means that
the model discrepancy did not exceed that empirical percentile;
`ABOVE_REFERENCE_RANGE` means that it did. Neither label is a hypothesis test,
an equality claim, or a judgement of physical acceptability.
`REFERENCE_UNRESOLVED` means that the reference width or its Monte Carlo
estimate is too unstable to support even this descriptive classification.

The gap is not a physical-severity scale, a probability, or a model score. Do
not average gaps, compare them across different sample sizes, or treat gap 1 as
a universal tolerance. Native-unit discrepancies and scientifically chosen
controls carry the substantive interpretation.

## Conditional generators

Inspect native conditions before any pooled result. If the real and generated
condition frequencies differ, the as-provided global comparison measures that
composition mismatch as well as field differences; GAUGE labels it as a
dataset-composition audit, not as conditional-model fidelity.

When there are at most eight conditions, the main profile shows every native
condition. For larger condition sets, the compact profile summarizes the
distribution across conditions while the exact condition-by-metric matrix is
exported for the appendix. A pooled tail group never replaces its component
conditions.

## Memorization

The localized copying audit compares each generated field's distance to nearby
training fields with the corresponding distance for held-out real fields,
within matched conditions and training-defined feature regions. A
generated-minus-real effect below zero is evidence consistent with localized
copying. Resolution is demonstrated with known copied fractions; failure to
resolve a small copied fraction is reported rather than converted into a
no-copy claim.
