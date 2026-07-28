# TC v0.2 analysis-protocol rationale

The machine-readable contract is
[`analysis_protocol.yaml`](analysis_protocol.yaml). This note explains its
status and the choices most likely to be misunderstood.

## Status

GAUGE and the TC example were developed iteratively, and the held-out archive
had already been examined before the submission-readiness revision. The v0.2
document is therefore a versioned analysis protocol with an amendment log, not
an untouched prospective preregistration. The empirical-reference contract,
six native conditions, metric definitions, controls, and secondary audits were
fixed before the clean v0.2 rerun.

## Data population

- All labels and continuous conditions are IBTrACS `USA_WIND` one-minute
  sustained winds.
- Train, validation, and test are storm disjoint.
- The default population gives every eligible storm equal weight and then every
  frame within that storm equal conditional weight.
- Primary results use TS and Categories 1–5 separately. There is no C4+ pooled
  fidelity family.
- The naturally imbalanced real mixture and class-balanced generated bank are
  compared globally only as a dataset-composition audit.

## Representation

Raw arrays remain checksum preserved. All metric inputs use the generator's
declared 170–325 K clipping representation. Those endpoints are training
preprocessing quantiles, not physical limits; contact with them is reported as
a preprocessing saturation audit.

The radial PSD removes each field's spatial mean, applies a separable Hann
window, divides power by window energy, excludes the zero mode, and uses 24
nonempty discrete-support-aware radial wavenumber bins. Its scalar discrepancy
is mean squared log10-power difference in `dex²`.

## Empirical reference

Each metric and native condition is evaluated against event-disjoint real
subsets with matching sample size and weighting. The stored q50 and q95 are
finite-archive reference summaries, not hypothesis-test thresholds. Their
paired-bootstrap Monte Carlo intervals and split-half stability are reported.
Unstable or degenerate references fail closed as `REFERENCE_UNRESOLVED`.

The seven functional W1 results use 2,000 repetitions. Paper-facing curve and
feature metrics use 1,000, with 2,000 for coverage; exploratory dependence
summaries use 200. Quadratic feature estimators use a deterministic matched cap
of 256 fields per side and record this effective count.

## Controls and copying

Controls use reserved training storms only. Each metric has a specific
sensitivity or invariance control and reports its complete severity response,
monotonicity, and first empirical-reference crossing. A failed control withholds
affirmative interpretation but does not hide the raw result.

The memorization representation is training fitted. The nearest-neighbour
reference includes every diffusion-training field, while the EOF basis,
standardisation, and MMD bandwidth use their separately declared training-only
fit sets. Regional copying summaries are descriptive central ranges, not
multiplicity-adjusted tests. Resolution is shown with copied fractions from 1%
to 100%.

## Secondary audits

The class-conditional comparison remains primary. A descriptive continuous-wind
audit estimates within-class functional slopes: real fields receive inverse
storm-frequency weights and storm-bootstrap intervals, whereas generated fields
receive equal weights and sample-bootstrap intervals conditional on the fixed
bank. Synthetic responsive and wind-ignoring generators validate the slope
engine. No equivalence or fidelity verdict is attached to this audit.

Feature-space findings are also repeated post hoc across EOF fit sizes 256, 512,
and 1024, ranks 10, 20, and 40, and training-fitted low-pass representations at
16, 32, and 64 km. The registered rank-20, 512-field representation remains the
primary analysis; the variants reveal whether a qualitative pattern depends on
one reasonable representation choice.

## Remaining external gates

Independent statistical review, permission-cleared permanent artifact
deposition, and clean-room reproduction remain open. Literature-review
reconstruction is a separate user-owned workstream.
