# Statistical contract and assumptions

GAUGE v0.2 provides empirical reference calibration for unpaired generated
fields. It does not provide a hypothesis test, p-value, population-equivalence
claim, physical tolerance, or aggregate model score.

## Real-reference distribution

For each registered metric and condition, GAUGE repeatedly divides the eligible
held-out real dependence blocks into event-disjoint halves. One half supplies a
pseudo-generated real sample and the other supplies the comparison sample.
Both use the declared evaluation size and weighting estimand. The default
block-balanced estimand samples a block uniformly and then a frame within that
block; the optional frame-weighted estimand samples frames directly.

The same comparison half is used for the corresponding fixed-bank
generated-versus-real discrepancy. When the requested evaluation size equals
the generated-bank size, every repetition uses the complete generated bank.
When it is smaller, generated samples are drawn without replacement.

The resulting real-reference distribution describes finite-archive variation
under this exact design. It is not a noise floor, oracle, Bayes error, or sample
from a formal exchangeable null distribution.

## Reference gap and range labels

Let \(D_M\) be the median generated-versus-real discrepancy over admissible
reference repetitions, and let \(q_{50}^{RR}\) and \(q_{p}^{RR}\) be the median
and selected upper percentile of the real-reference discrepancies. The default
is \(p=0.95\), for which GAUGE reports

\[
g = \frac{D_M-q_{50}^{RR}}{q_{95}^{RR}-q_{50}^{RR}}.
\]

Thus, gap zero marks the typical real-reference discrepancy and gap one marks
its selected upper percentile (the 95th percentile by default). This common
visual coordinate is conditional on the
archive, sample size, block definition, condition, metric, and weighting rule.
It cannot be averaged across metrics or interpreted as physical severity.

- `WITHIN_REFERENCE_RANGE`: \(D_M \le q_{p}^{RR}\).
- `ABOVE_REFERENCE_RANGE`: \(D_M > q_{p}^{RR}\).
- `REFERENCE_UNRESOLVED`: the reference width or numerical stability is
  inadequate.
- `NOT_EVALUATED`: the metric or design is invalid for the supplied data.

These labels are descriptive. GAUGE makes no joint verdict across a collection
of metric-by-condition results and prohibits counting range labels as though
they were independent significance tests.

## Two distinct uncertainty summaries

The native-unit comparison range is the central configured interval of the
generated-versus-real discrepancies obtained across admissible real subsets.
It includes variation from real-block partitioning and event/frame sampling,
plus generated subsampling when the supplied bank is larger than the declared
evaluation size. It does not include generator retraining, a newly generated
bank, or an unobserved population of storms.

Separately, GAUGE uses paired bootstrap resampling of the stored
real-reference/model-reference repetitions to report conditional Monte Carlo
intervals for \(q_{50}^{RR}\), \(q_{95}^{RR}\), and the central gap. These
intervals quantify numerical precision conditional on the fixed archive and
generated bank. Archive sensitivity requires a delete-block or comparable
case-specific analysis and is not hidden inside the Monte Carlo interval.
When a non-default reference quantile is selected, the report retains the fixed
\(q_{95}^{RR}\) interval for comparability and separately records the interval
for \(q_{p}^{RR}\), which is the boundary used in the gap.

## Metric controls

Every built-in paper-facing metric declares sensitivity or invariance controls.
Controls are derived from training data or explicitly reserved real blocks and
never from held-out model outcomes.
Feature-space controls reuse the exact training-fitted representation,
standardisation, and kernel parameters used by the reported feature metric.

- `CONTROL_VALIDATED`: expected sensitivity and invariance contracts hold, and
  a registered positive control reaches the real-reference range.
- `RESPONSIVE_BELOW_REFERENCE_RANGE`: the metric responds monotonically, but
  the registered severity range does not reach the reference marker.
- `CONTROL_FAILED`: an expected response or invariance fails.
- `CONTROL_NOT_REGISTERED`: no applicable control was supplied.

Raw discrepancies and gaps remain visible after a control failure, but GAUGE
withholds affirmative interpretation. Controls describe what the evaluation can
resolve; their severities are not universal acceptability thresholds.

## Scope and responsibility

Version 0.2 targets unpaired sample-to-population evaluation of independent
single- or multichannel two-dimensional gridded fields. Paired forecasts,
downscaling, trajectories, temporal coherence, and coupled-variable constraints
require complementary registered verification.

Dependence blocks must represent the scientific sampling unit. GAUGE checks
overlap, missing identifiers, and minimum counts, but cannot infer whether a
syntactically valid identifier is scientifically appropriate. Validation
therefore demonstrates how incorrectly treating dependent frames as independent
changes the reference distribution.
