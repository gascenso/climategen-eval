# GAUGE v0.2 scientific blueprint

**GAUGE** (*Guidelines for the Assessment and Uniform reporting of Generative
Earth-system models*) is an empirical reference-calibration framework for
unpaired generated geophysical fields. It combines native-unit physical
diagnostics, an event-aware real-versus-real reference, controlled degraded
data, diversity and copying audits, and a compact paper-facing report.

This is the canonical v0.2 scientific contract. It supersedes the earlier
testing-oriented blueprint.

## 1. Problem, contribution, and scope

No single image or distribution metric establishes the physical fidelity of a
climate generative model. Computer-vision scores can be physically opaque;
unreferenced histograms and spectra do not show how much finite real samples
disagree; global summaries can hide conditional failures; and feature distances
can miss mode loss or localized training replay.

GAUGE addresses this interpretation problem with four linked components:

1. scalar physical functionals reported through W1 in native units;
2. an empirical real-reference distribution matched to sample size,
   conditioning, event dependence, and weighting;
3. metric-specific sensitivity, invariance, and copying controls;
4. a layered report that makes the main result compact while preserving every
   raw value, warning, and provenance record.

Version 0.2 covers unpaired sample-to-population fidelity for independent
single- or multichannel two-dimensional gridded fields. It does not replace
CRPS, reliability diagrams, rank histograms, FSS, SAL, or other methods for
paired forecasts, downscaling cases, trajectories, or coupled systems. Those
questions require complementary verification or future adapters.

GAUGE does not certify physical correctness, define a universal tolerance, or
rank models through an aggregate score.

## 2. Data contract and population represented

Inputs are:

- `train`: real training fields used only to fit representations and controls
  and to provide the training side of the copying audit;
- `real`: held-out real fields with scientifically meaningful independent-event
  identifiers;
- `generated`: a fixed generated sample bank, with conditions when applicable;
- scalar physical functionals and optional domain metrics;
- an evaluation configuration and optional registered controls.

All fields from one physical event must stay in one data partition. An event
identifier may denote a storm, day, simulation member, or another unit justified
by the data-generating process. GAUGE validates identifiers and overlap but
cannot infer whether the user's chosen event is scientifically correct.

The default population is **block balanced**: select an event uniformly, then a
field uniformly from that event. This prevents long or densely observed events
from dominating the result. The optional frame-weighted population selects a
field uniformly. Generated, pseudo-generated-real, and real-reference samples
use the same declared sample count and weighting policy.

For conditional models, native-condition results are primary. If real and
generated condition proportions differ, their as-provided global comparison is
a dataset-composition audit, not a statement of conditional fidelity.

## 3. Registered evidence

### 3.1 Scalar physical functionals

An application registers approximately five to ten scientifically motivated
scalar summaries spanning amplitude, spatial organisation or scale, and at
least one important tail. Each functional has:

- an identifier and plain-language definition;
- physical units and valid support;
- preprocessing and mask rules;
- vocabulary for directional differences.

GAUGE compares real and generated functional distributions with exact
one-dimensional Wasserstein-1 distance. Its unit is the functional's unit. W1
is a common reporting form only for such scalar functionals; it is not a
universal distance for fields, curves, or multivariate processes.

### 3.2 Spatial and feature diagnostics

Applications may register curve distances such as radial profiles or spectra.
Each curve metric fixes its coordinates, masks, detrending, window, normalising
convention, binning, and missing-value policy.

The core feature registry contains:

- non-negative biased/V-statistic Gaussian-kernel MMD²;
- density departure;
- coverage deficit;
- two-sided diversity departure.

Standardisation, EOF/PCA bases, kernels, and bandwidths are fitted on declared
training data only. A user-provided embedding is permitted when its training
provenance and valid domain are recorded. Feature metrics are useful
complements but do not acquire physical units from the reference scaling.
Feature-space controls must reuse the exact fitted representation and kernel
parameters used by the reported model comparison; fitting a second
control-specific representation would validate a different metric.

### 3.3 Localized copying

The copying audit partitions queries jointly by condition and a
training-defined feature region. Within supported partitions it compares
generated-to-training nearest-neighbour distances with
held-out-real-to-training distances. Resampling preserves held-out event
dependence.

The result is interpreted only after an exact-copy positive control succeeds.
A copied-fraction curve at 1%, 5%, 10%, 25%, 50%, and 100% reports the smallest
mixture resolved by this specific archive and representation. Failure to flag a
model is therefore not described as proof that no training sample was copied.

### 3.4 Preprocessing saturation

For the TC adapter, the former “physical admissibility” row is specifically a
**preprocessing saturation audit**: it measures excess contact with the
declared clipping boundaries. It is not a general physical-validity test.
Other applications may register genuinely physical constraints under their own
explicit definitions.

## 4. Empirical real-reference calibration

For one metric and one condition, GAUGE repeatedly:

1. splits eligible held-out real events into disjoint halves;
2. draws \(N\) pseudo-generated real fields from one half under the declared
   weighting policy;
3. draws \(N\) reference real fields from the other half under the same policy;
4. evaluates their discrepancy \(D_k^{RR}\);
5. compares the fixed generated bank with the same reference draw, producing
   \(D_k^{MR}\).

The stored \(\{D_k^{RR}\}\) values form the **real-reference distribution**.
They describe finite-archive discrepancies under this exact sample size,
conditioning, event partition, preprocessing, and weighting. They are not a
noise floor, oracle, Bayes error, population-equality distribution, or exact
level-\(\alpha\) null.

For each result GAUGE reports:

- the median model-versus-real discrepancy and its central comparison range
  across admissible reference subsets;
- real-reference \(q_{50}\), \(q_{95}\), and supporting \(q_{2.5}\) and
  \(q_{97.5}\);
- conditional Monte Carlo intervals for \(q_{50}\) and \(q_{95}\);
- the empirical percentile of the model discrepancy;
- sample and event counts, estimand, warnings, and provenance.

Summary Monte Carlo uncertainty is obtained by paired bootstrap resampling of
the stored real-reference and model-reference repetitions. It quantifies the
numerical stability of the finite archive calculation. It does not include
uncertainty from a different real archive, retraining the generator, or a
physical tolerance.

## 5. The descriptive real-reference gap

GAUGE uses the compact plotting coordinate

\[
g=\frac{D_{\mathrm{model}}-q_{50}^{RR}}
        {q_{95}^{RR}-q_{50}^{RR}}.
\]

Here \(g=0\) marks the median event-disjoint real-versus-real discrepancy and
\(g=1\) its empirical 95th percentile. This makes heterogeneous metrics
visually comparable within the declared evaluation design. Degraded-data
controls are placed on the same axis as recognizable signposts.

The gap is descriptive. It is not physical severity, a p-value, a probability
of realism, or a model score. It must not be averaged across metrics or compared
across sample sizes. A high gap can identify an unusual finite-sample
discrepancy but does not specify its application consequence; the native-unit
value and diagnostic definition remain authoritative.

## 6. Result and control statuses

Each metric-by-condition result receives one empirical-reference label:

- `WITHIN_REFERENCE_RANGE`: the model discrepancy is at or below the registered
  real-reference quantile;
- `ABOVE_REFERENCE_RANGE`: it is above that quantile;
- `REFERENCE_UNRESOLVED`: the reference width or Monte Carlo stability is
  inadequate;
- `NOT_EVALUATED`: the metric is undefined, unsupported, or has invalid input.

These are presentation labels, not hypothesis-test decisions. No familywise,
Bonferroni, simultaneous, global pass/fail, or “number of failed cells” verdict
is produced.

Control interpretation is tracked separately:

- `CONTROL_VALIDATED`;
- `RESPONSIVE_BELOW_REFERENCE_RANGE`;
- `CONTROL_FAILED`;
- `CONTROL_NOT_REGISTERED`.

Raw results remain visible after a control failure, but the report withholds an
affirmative interpretation for that metric.

## 7. Metric-specific controls

Controls are built from reserved training events and declared before the clean
analysis run. Each control records its valid metrics, expected direction,
severity curve, monotonicity, first boundary-crossing severity, and any
invariance requirement.

The TC registry contains:

- additive temperature shifts for core-minimum and lower-tail temperature;
- warming near 220 K for cold-cloud fraction;
- eye/eyewall perturbations for eye contrast;
- displacement, blur, and radial deformation for radial diagnostics;
- phase randomisation for spatial organisation while preserving the unwindowed
  Fourier-power magnitude;
- blur and a high-frequency artifact as positive PSD controls, plus a
  quarter-turn rotation as a radial-PSD invariance control;
- training-EOF excursions for MMD² and density;
- mode removal for coverage;
- contraction and expansion for diversity;
- boundary spikes for preprocessing saturation;
- increasing copied-training fractions for memorization resolution.

A control says whether a metric responds to a known change under the declared
design. It does not define a universal acceptable error or replace scientific
interpretation.

## 8. Validation before application

The generic harness evaluates the behaviour of the entire calibration
procedure, using the following neutral language:

- **same-population boundary-crossing rate** for repeated samples from the same
  synthetic population;
- **perturbation crossing probability** for known changes.

Validation covers:

- Gaussian, heavy-tailed, multimodal, shifted, narrowed, widened, and
  tail-contaminated scalar distributions;
- curve MAE and log-PSD discrepancy under known spatial-scale changes;
- MMD², density, coverage, and diversity under mode removal, duplication,
  contraction, expansion, and off-manifold contamination;
- balanced and severely unequal event sizes, 20- and 22-event thin conditions,
  multiple bank sizes, and train/test shifts;
- correct event identifiers versus intentionally wrong frame-IID identifiers;
- conditional Monte Carlo stability and grouped delete-event archive
  sensitivity;
- degenerate widths and unstable quantiles that fail closed.

Accepted behaviour requires same-population crossing not to exceed the nominal
5% tail beyond Monte Carlo uncertainty, monotone response curves, increasing
crossing for increasing perturbations, visible loss of resolution in thin-event
settings, non-negative MMD² up to numerical precision, and unresolved outputs
for degenerate or unstable references.

## 9. Reporting hierarchy

The main output is a polished six-condition gap profile containing:

- model gaps with central comparison ranges across admissible held-out-real
  subsets;
- all native intensity conditions;
- distinct grayscale-safe control signposts;
- no global mixture row and no aggregate score.

The accompanying native-unit table reports the full values, reference
quantiles, percentiles, counts, statuses, and warnings. A small number of
physical distributions and curves explain the most important pattern.

The appendix contains:

- the complete condition-by-metric gap matrix;
- all control response and invariance curves;
- all-class physical distributions and curve summaries;
- real-reference and gap stability;
- extended synthetic validation;
- EOF fit-size/rank and multiscale robustness;
- continuous-wind response;
- copied-fraction resolution;
- preprocessing, partition, sample-count, and provenance audits.

Every number, table, and figure is generated from schema-v3 artifacts.

## 10. TC demonstration

The case study evaluates a **TC brightness-temperature diffusion model**
(`baseline_ztsnr` only in reproducibility metadata). It generates
storm-centred \(256\times256\) infrared brightness-temperature fields at
continuous one-minute maximum-wind conditions.

Primary fidelity results use the six native TS and Saffir–Simpson Category 1–5
conditions. Category 5 remains separate and its 22-storm limitation is reported.
There is no C4+ pooled evaluation family. The unmatched global result is retained
only as a composition audit and excluded from the headline profile.

Seven physical functionals cover core temperature, cold-cloud area, radial
gradient location, azimuthal departure, deviation-angle variance, eye contrast,
and the lower BT tail. Radial-profile, log-PSD, training-fitted feature,
localized-copying, and preprocessing-saturation diagnostics provide
complementary views.

A secondary continuous-wind audit estimates within-class
functional-versus-wind slopes. Real slopes use inverse storm-frequency weights
and storm bootstrap intervals; generated slopes use equal sample weights and
sample bootstrap intervals. This audit has no equivalence label and does not
replace class-conditional fidelity.

EOF fit sizes 256/512/1024, ranks 10/20/40, and training-fitted low-pass EOF
representations at declared physical scales form a post-hoc robustness analysis.
Its status is stated explicitly.

## 11. Public interface and versioning

```python
import gauge_climate

train = gauge_climate.Dataset(fields, blocks=events, labels=labels)
real = gauge_climate.Dataset(test_fields, blocks=test_events, labels=test_labels)
generated = gauge_climate.Dataset(samples, labels=generated_labels)

report = gauge_climate.evaluate(
    train=train,
    real=real,
    generated=generated,
    functionals=functionals,
    metrics=metrics,
    config=config,
)
```

The distribution is `gauge-climate` v0.2.0, the import namespace is
`gauge_climate`, and the report schema is
`gauge_climate.evaluation.v3`. Schema-v2 reports can be read and migrated for
inspection, but they do not acquire new Monte Carlo quantities that were not
stored.

## 12. Analysis provenance and permitted claims

The TC archive informed framework development. The repository therefore uses a
versioned **analysis protocol and amendment log**, not a prospective
preregistration claim. The revised method was frozen before the clean v0.2
rerun, and all post-hoc robustness analyses are labelled as such.

GAUGE may be described as:

- a portable empirical reference-calibration and reporting framework for
  unpaired generated 2-D climate fields;
- a way to combine native-unit physical evidence with finite-archive context
  and recognizable degraded-model signposts;
- an approach that exposes conditional, diversity, copying, preprocessing, and
  resolution limitations without a universal score.

It must not be described as:

- an exact hypothesis test or population-equivalence procedure;
- a universal physical-realism certificate;
- a replacement for paired forecast or spatial verification;
- a framework with a universal tolerance or normalised 0–1 quality score;
- evidence that a model is “best” merely because its gap lies within the
  empirical reference range.

An archival release still requires permanent code/data deposition, independent
review, and clean-room reproduction.
