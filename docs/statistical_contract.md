# Statistical contract and assumptions

GAUGE v0.1 deliberately separates finite-N detection from application-tolerance
inference.

## Detection

For each metric, repeated whole-block splits create a pseudo-generated sample
and a reference-real sample with the same declared estimand and `N_eval`. The
model statistic is the median across matched model/reference draws and detection
uses the one-sided `(1 - alpha)` real-reference boundary. The procedure controls
false positives conservatively in the shipped Monte Carlo regimes; it is not
claimed to attain alpha exactly. Position relative to this envelope is a
finite-N detectability diagnostic, not evidence of population equivalence.

Joint reference claims use Bonferroni-adjusted cell-wise order-statistic
boundaries. This controls the registered family without pretending that
conditional strata have a valid shared resampling coupling. A family of `M`
cells requires at least `M / alpha - 1` reference resamples to resolve the
adjusted empirical tail; otherwise GAUGE withholds the simultaneous conclusion.
A family is never inferred from the rows shown in a report: its exact
`stratum:metric` keys must be preregistered in `simultaneous_family`; the empty
default makes no joint claim.
A less conservative max-statistic method may be registered only when every cell
genuinely shares resampling units and draws.

The resampling budget is part of the analysis contract. Functional W1 cells and
every cell in a preregistered simultaneous family use `n_resamples`. Secondary
metrics without tolerance conclusions may use the smaller
`diagnostic_resamples` budget (default `min(n_resamples, 200)`), which is reported
in every cell. Quadratic feature-set diagnostics may likewise declare a matched
per-draw sample cap. These choices affect Monte Carlo precision and diagnostic
resolution, not the target estimand, and must be fixed before model outcomes.
Complete reference plans are deterministic from inputs, configuration, and seed;
reports retain a SHA-256 digest and preview instead of expanding every index.

For a conditional generator, the distribution of conditions is part of the
target. GAUGE audits real and generated label proportions whenever conditional
rows are requested. An as-provided global comparison with different proportions
is a dataset-composition test, not a conditional-model fidelity test. It remains
reportable, but must be labeled as such; a global conditional-fidelity claim
requires a predeclared common condition-mixture policy.

## Bounded scalar W1 confidence intervals

For a scalar functional with finite support `[a, b]`, the block-balanced target
is the distribution obtained by drawing a population block uniformly and then a
frame uniformly within it. The empirical real CDF is therefore the average of
the within-block CDFs. Every frame informs its block's CDF, but the independent
population replication remains the number of blocks. Generated fields are
assumed mutually independent.

In one dimension,

```text
W1(F, G) = integral |F(t) - G(t)| dt.
```

Away from equality, its derivative integrates CDF perturbations against
`sign(F-G)`. GAUGE evaluates that derivative exactly over the pooled empirical
support. One influence value is formed for each whole real block and one for
each generated field. Their variance components are combined and a
Welch--Satterthwaite t interval is reported. This is the clustered analogue of
the two-sample alternative central limit theorem for empirical W1; asymptotic
normality for Wasserstein-type distances between distinct laws is established
by [Berthet, Fort & Klein (2020)](https://arxiv.org/abs/1710.09763).

The assumptions are explicit: independent and identically sampled population
blocks, independent generated fields, finite second moments (bounded support in
v0.1), and a regular nonzero tolerance boundary where the population CDFs do
not coincide over a set of positive Lebesgue measure. The interval is two-sided
`1-alpha`; requiring its upper endpoint below the tolerance is therefore
slightly conservative relative to a standalone one-sided alpha-level test.
The shipped Monte Carlo gate checks equality, boundary shifts, crossing CDFs,
tail contamination, 22-block strata, and severely unequal block sizes.

Functionals registered as discrete or nonregular use the finite-sample DKW
quantile-band interval of [Manole, Balakrishnan & Wasserman
(2022)](https://doi.org/10.1214/22-EJS2001). This fallback is robust but may be
underpowered. Without a defensible finite support, or under the frame-weighted
estimand, GAUGE reports a whole-block percentile bootstrap only as a diagnostic
and withholds tolerance conclusions. The classical bootstrap is not promoted
to a formal W1 interval because it can fail at important nulls; see
[Sommerfeld & Munk (2018)](https://doi.org/10.1111/rssb.12236). No clipping is
performed merely to force values into a declared support.

## Permitted practical wording

Shift sensitivity is evaluated at the exact application tolerance in addition
to the generic severity grid. Power at that point is a design diagnostic; it is
not a model result and cannot establish equivalence.

- `WITHIN_TOLERANCE`: the direct upper W1 bound is below the frozen native-unit
  tolerance.
- `EXCEEDS_TOLERANCE`: the direct lower bound is above it.
- `INCONCLUSIVE`: the interval crosses the tolerance, but its uncertainty
  margin is smaller than the tolerance.
- `UNDERPOWERED`: the interval crosses the tolerance and its uncertainty margin
  alone is at least as large as the tolerance, or the stratum fails the declared
  minimum-block requirement.

These conclusions require a tolerance justified independently of final test
results. Shift sensitivity remains a separate detection diagnostic and never
changes a practical verdict. A failure to cross the real-reference boundary
never substitutes for the direct upper-bound condition.
