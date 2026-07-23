# Statistical contract and assumptions

GAUGE makes one inferential claim: whether a registered discrepancy exceeds
the finite-sample variability observed between dependence-matched real samples
at the declared evaluation size. It does not infer a universal application
equivalence threshold.

## Pointwise detection

For each metric and stratum, repeated whole-block splits create a
pseudo-generated sample and a reference-real sample under the same estimand and
`N_eval` used for the model comparison. The model statistic is the median across
matched model/reference draws. Detection uses the one-sided `(1 - alpha)`
real-reference boundary.

- `DEPARTURE_DETECTED`: the model statistic exceeds the boundary.
- `NO_DEPARTURE_DETECTED`: it does not.
- `NOT_EVALUATED`: the design is invalid, too thin, unstable, or failed a
  registered sensitivity control.

The procedure is conservative in the shipped Monte Carlo regimes; it is not
claimed to attain `alpha` exactly. Non-detection is not evidence of equality,
physical realism, or application acceptability.

## Raw discrepancy intervals

The printed raw interval is the central `(1 - alpha)` interval of the matched
model-versus-reference resampling distribution. It quantifies variation induced
by whole-block real resampling, generated subsampling when `N_eval` is smaller
than the supplied bank, and the declared finite evaluation design. When
`N_eval` equals the bank size, the generated bank is intentionally fixed across
draws: the interval is conditional on that bank and does not estimate
generator-population uncertainty. It is not presented as a regular
population-W1 equivalence interval.

For scalar functionals, W1 and its interval remain in the functional’s native
units. Curve, feature, rate, and dependence metrics retain their registered
scales. The same resampling plan backs the raw interval, reference envelope,
and gap uncertainty.

## Simultaneous claims

Joint reference claims use Bonferroni-adjusted cell-wise order-statistic
boundaries. This controls a registered family without assuming that conditional
strata share a common block-resampling coupling. A family of `M` cells requires
at least `M / alpha - 1` reference resamples to resolve the adjusted empirical
tail; otherwise GAUGE withholds the simultaneous conclusion.

The exact `stratum:metric` keys must be preregistered in
`simultaneous_family`; the empty default makes no joint claim. A less
conservative max-statistic method is valid only when every cell genuinely shares
resampling units and draws.

## Resampling and conditioning contract

Functional W1 cells and all cells in a simultaneous family use `n_resamples`.
Secondary diagnostics may use the smaller `diagnostic_resamples` budget.
Predeclared headline diagnostics may receive a larger
`diagnostic_resamples_by_metric` budget without being added to a simultaneous
inferential family. The effective count is reported per cell. A cell whose
reference boundary fails the declared split-half stability check retains its raw
estimate and gap for audit but receives `NOT_EVALUATED`, not a categorical
detection conclusion. Quadratic feature metrics may use a declared matched
sample cap. Complete reference plans are deterministic from the inputs,
configuration, and seed; reports retain a SHA-256 digest and preview.

For a conditional generator, the condition distribution is part of the target.
An as-provided global comparison with unequal label proportions is a composition
test, not a conditional-fidelity test. A global fidelity claim requires a
predeclared common mixture; otherwise the matched conditional rows carry the
claim.

## Sensitivity resolution

For scalar functionals, GAUGE reports the smallest evaluated uniform
native-unit shift detected with the registered target probability. Other
metrics use predeclared corruption families. These are design-sensitivity
diagnostics: they establish what failures a metric can detect at this `N`, not
which discrepancies are universally acceptable.
