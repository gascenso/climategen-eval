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

## Bounded scalar W1 confidence bounds

For a scalar functional with finite support `[a, b]`, the block-balanced target
is the distribution obtained by drawing a population block uniformly and then a
frame uniformly within it. GAUGE draws one frame uniformly from every held-out
block. If blocks are independent and identically sampled, these values are an
i.i.d. sample from that target; generated fields are assumed mutually independent
in v0.1.

For sample sizes `n` and `m`, GAUGE uses simultaneous Dvoretzky–Kiefer–Wolfowitz
CDF bands with

```text
beta_n = sqrt(log(4 / alpha) / (2 n))
beta_m = sqrt(log(4 / alpha) / (2 m)).
```

The corresponding empirical quantile bands are combined pointwise and integrated
over `[0, 1]` to bound the untrimmed W1 distance. The known support supplies the
quantiles outside the empirical probability range. This follows the finite-sample
quantile-band construction in [Manole, Balakrishnan & Wasserman
(2022)](https://doi.org/10.1214/22-EJS2001) and is conservative by design.

The method discards within-block replication for its formal bound. Those frames
still inform the real-reference detector and diagnostics, but they are not
pretended to be independent evidence. This cost in precision is an explicit
v0.1 limitation and a target for later cluster-empirical-process work.

Without a defensible finite support, or under the frame-weighted estimand, GAUGE
reports a whole-block clustered percentile bootstrap only as a diagnostic and
withholds `WITHIN_TOLERANCE`. Classical empirical-Wasserstein bootstrap behavior
can fail at important nulls; see, for example, [Sommerfeld & Munk
(2018)](https://doi.org/10.1111/rssb.12233). No clipping is performed to force a
functional into its declared support.

## Permitted practical wording

- `WITHIN_TOLERANCE`: the direct upper W1 bound is below the frozen native-unit
  tolerance.
- `EXCEEDS_TOLERANCE`: the direct lower bound is above it.
- `INCONCLUSIVE`: the interval crosses the tolerance despite adequate declared
  shift resolution.
- `UNDERPOWERED`: it crosses the tolerance and the sensitivity design cannot
  resolve a shift of that size.

These conclusions require a tolerance justified independently of final test
results. A failure to cross the real-reference boundary never substitutes for
the direct upper-bound condition.
