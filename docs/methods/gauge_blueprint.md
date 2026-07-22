# GAUGE — scientific and implementation blueprint

**GAUGE** (*Guidelines for the Assessment and Uniform reporting of Generative
Earth-system models*) is a reference-aware protocol for evaluating generated
geophysical fields. It turns a bank of generated samples, held-out real data,
and training data into a compact paper-ready summary backed by reproducible
uncertainty, sensitivity controls, and complete provenance.

Working title: **“GAUGE: reference-aware evaluation of generative models for
climate fields.”**

This document is the canonical scientific contract. Historical development
plans are not part of the protocol.

## 1. Thesis, scope, and user promise

### 1.1 The problem

No single metric establishes the physical realism of generated climate or
weather fields. Image-quality scores can be physically opaque; marginal
histograms can hide spatial or conditional failures; diversity summaries can
miss localized copying; and raw discrepancies are difficult to interpret
without knowing how much two finite real samples disagree.

GAUGE addresses this by combining four ideas:

1. physically interpretable scalar functionals in native units;
2. a dependence-matched real-reference envelope at the declared sample size;
3. complementary structure, diversity, memorization, and admissibility checks;
4. predeclared degraded-data controls showing that registered metrics respond
   to failures they are supposed to detect.

### 1.2 Scope

Version 1 supports independent generated samples consisting of 2-D
single-channel or multichannel gridded fields. Dependence among held-out real
samples is represented through user-supplied blocks such as storms, days, or
simulation members. Adapters may add domain-specific functionals and curve
metrics.

Arbitrary spatiotemporal trajectories, coupled multivariate systems, and
probabilistic forecasts require specialized extensions. When a generator is
itself a forecast, established probabilistic verification remains authoritative
for calibration and forecast skill.

GAUGE evaluates intrinsic fidelity. It does not claim that generated samples
improve every possible use, certify a model as physically correct, or produce a
universal model ranking.

### 1.3 Zero-configuration contract

The standard workflow must not require users to invent statistical thresholds:

```python
report = gauge_climate.evaluate(
    train=train,
    real=real,
    generated=generated,
    functionals=functionals,
)
```

With scientifically named functionals and correctly declared dependence
blocks, the default report includes native-unit discrepancies and uncertainty,
the real-reference envelope, a reference-calibrated gap profile, condition
audits, registered controls, diversity, memorization, admissibility, and full
provenance. The command-line equivalent is one `gauge-climate run` invocation
followed by `gauge-climate report`.

GAUGE deliberately does not ask users to invent application thresholds. It
reports what departure was measured, how uncertain it is, where it lies
relative to finite-real variability, and which controlled failures the metrics
can detect. It makes no universal application-equivalence claim.

## 2. Data contract and estimand

Inputs are:

- `X_train`: real training fields, used only for fitting features, controls,
  and the memorization reference;
- `X_real`: held-out real fields, with dependence-block identifiers;
- `X_gen`: generated fields, with labels when the generator is conditional;
- `functionals`: scalar domain quantities with names, units, definitions, and
  valid input domains;
- optional domain adapters, strata, and controls.

No statistic, embedding, normalization, threshold, or control may be fitted to
generated outcomes or held-out real outcomes unless its role explicitly
requires held-out comparison and the operation is recorded.

The default target is the **block-balanced distribution**: sample a block
uniformly and then sample a frame uniformly within that block. This prevents a
long storm or densely sampled period from silently dominating the population.
An optional frame-weighted estimand samples frames uniformly. Pseudo-generated,
reference, and generated comparisons must use the same weighting policy and
evaluation sample size.

For a conditional generator, the distribution of conditions is part of the
target. If real and generated label proportions differ, an as-provided global
comparison is a composition check rather than a conditional-fidelity result.
Conditional rows or a predeclared common-mixture comparison support fidelity
claims.

## 3. Metric architecture

### 3.1 Physically interpretable functionals

Each application registers approximately 5–10 scalar functionals spanning:

- amplitude or intensity;
- spatial organization or scale;
- extremes or tails.

At least one functional should explicitly target an important tail. Every
functional has a physical definition, units, direction vocabulary, and valid
input domain. GAUGE uses exact one-dimensional Wasserstein-1 distance between
the real and generated functional distributions. W1 is a common currency only
across scalar, physically interpretable functionals; its native unit remains
the unit of the functional.

### 3.2 Complementary registered metrics

The core registry supports:

- functional W1;
- training-fitted feature MMD²;
- density, coverage, and two-sided diversity departure;
- localized memorization;
- hard and soft admissibility checks;
- dependence closure among functionals.

Radial profiles, radial power spectra, and similar curve comparisons are domain
adapters, not universal metrics. Every curve adapter fixes masks, detrending,
windowing, normalization, physical coordinates, binning, and missing-value
handling.

Feature standardization and optional EOF/PCA embeddings are fitted on training
data only. A user-supplied training-fitted embedding is permitted and its
provenance is recorded. No feature metric is presented as a physically
interpretable native-unit discrepancy.

### 3.3 Memorization

GAUGE fits the feature space and feature-space regions on training data only.
Within each occupied region, it compares generated-to-training nearest-neighbor
distances with held-out-real-to-training distances. When labels exist, the
comparison is partitioned jointly by label and region. Uncertainty resamples
generated queries and whole held-out blocks, with multiplicity control across
supported cells.

Generated queries significantly closer to training than held-out real queries
support a localized-copying conclusion. Significantly farther means only that
generated queries are farther from training; it is not automatically a
physical-admissibility failure. An exact training-copy control must trigger the
test before model results are interpreted.

## 4. Real-reference engine

The real-reference envelope estimates how much discrepancy arises when two
finite samples are both drawn from held-out reality under the declared
dependence structure and estimand.

For each registered metric and stratum:

```text
for k = 1,...,K:
    split eligible real blocks into disjoint halves A_k and B_k
    pseudo_k = draw N samples from A_k under the declared estimand
    real_k   = draw N samples from B_k under the same estimand
    gen_k    = draw N generated samples without replacement
    reference_k = metric(pseudo_k, real_k)
    model_k     = metric(gen_k, real_k)
```

Whole blocks are split and resampled; partial-block truncation is prohibited.
Every draw records its seed, effective sample and block counts, deterministic
partition digest, and preprocessing provenance. Functional cells and any
registered simultaneous family use the primary resampling budget. Secondary
diagnostics may use a smaller declared budget, which is recorded per cell.

The model statistic is compared with a one-sided real-reference boundary at the
declared `N`. A pointwise result is:

- `DEPARTURE_DETECTED` when the statistic exceeds the boundary;
- `NO_DEPARTURE_DETECTED` otherwise.

The latter is a failure to detect at this sample size, not proof of equality,
equivalence, or physical realism. Joint claims require an explicit registered
family and a valid simultaneous procedure; the default makes only cell-wise
claims.

## 5. Reference-calibrated gap

The primary visual coordinate is:

```text
gap = (model discrepancy - real-reference median) /
      (pointwise reference boundary - real-reference median)
```

Interpretation is intentionally simple:

- `gap = 0`: model discrepancy equals the median real-vs-real discrepancy;
- `gap = 1`: model discrepancy reaches the pointwise detection boundary;
- larger values: progressively farther beyond finite-real sampling variation.

The gap is a presentation and detectability coordinate. It does not express
physical severity, cannot be compared across sample sizes, cannot establish
equivalence, and must never be averaged into an aggregate model score. Every
gap is linked to its raw estimate and interval in native units.

Predeclared degenerate or corrupted-data controls share this coordinate. They
provide recognizable severity anchors—such as copying, blur, phase shuffling,
wrong-condition sampling, or low-order climatology—without pretending that the
anchors define universal acceptability.

## 6. Sensitivity controls

Each metric declares the type of failure it should detect and a predeclared
perturbation family. Examples include:

| Control | Intended diagnostic |
|---|---|
| exact training replay | localized memorization |
| uniform physical-unit shift | amplitude functionals |
| Gaussian blur sweep | gradients and small-scale spectrum |
| phase shuffle | organization with Fourier-power preservation |
| center jitter | radial structure |
| wrong-condition samples | conditional fidelity |

For ordered severity sweeps, GAUGE reports response direction, monotonicity, and
whether the frozen maximum severity crosses the reference boundary. Binary
positive controls require their declared response. A metric failing its control
is marked uninterpretable for the affected claim.

Controls answer “does this metric respond to a known failure at this `N`?” They
do **not** define universal application acceptability. GAUGE reports that limit
plainly instead of converting a control severity or sampling boundary into a
fabricated equivalence threshold.

## 7. Conditional reporting without figure explosion

The main figure must not let a frequent class hide failures in rare or important
conditions.

- For `K <= 8`, show every native condition and every predeclared priority
  aggregate in one stratified gap profile.
- For `K > 8`, show the native-condition median and range, label the largest
  observed gap, and retain all predeclared priority conditions.
- Always export the exact condition-by-metric matrix and native-unit table to the
  appendix bundle.

“Largest observed” is descriptive unless a worst-condition inferential claim
was registered. Pooled strata never replace their component rows and may not be
invented after results are inspected.

## 8. Interpretation-first output

The default output is a paper bundle generated from the same versioned report
object as the machine-readable JSON:

1. one polished reference-gap profile with uncertainty and selected controls;
2. one native-unit table with raw effects, intervals, sample/block counts, and
   detection status;
3. a small number of physical distribution or curve drill-downs;
4. a complete conditional appendix matrix;
5. a technical audit containing sensitivity, memorization, preprocessing,
   warnings, software versions, seeds, and artifact checksums.

The main text can therefore remain visually simple without discarding the
statistical machinery needed for review and reproduction. GAUGE produces no
traffic-light certification and no aggregate score.

Each report cell contains:

- metric and stratum;
- raw estimate and uncertainty interval;
- units and orientation;
- real-reference median and boundary;
- reference-calibrated gap and its uncertainty;
- sample and block counts;
- weighting estimand;
- detection status;
- sensitivity resolution and control results;
- complete provenance.

## 9. Public interface

```python
import gauge_climate

train = gauge_climate.Dataset(fields, blocks=storms, labels=labels)
real = gauge_climate.Dataset(test_fields, blocks=test_storms, labels=test_labels)
generated = gauge_climate.Dataset(samples, labels=generated_labels)

report = gauge_climate.evaluate(
    train=train,
    real=real,
    generated=generated,
    functionals=functionals,
    metrics=metrics,       # optional registry override
    config=config,         # optional advanced configuration
)

report.write_json("gauge-report.json")
report.write_paper_bundle("paper-results")
```

Stable public types are `Dataset`, `FunctionalSpec`, `MetricSpec`,
`EvaluationConfig`, and `EvaluationReport`. NumPy and memory-mapped arrays are
core inputs; xarray is optional and Dask materialization is controlled and
recorded. Validation rejects overlapping blocks, missing identifiers, label
misalignment, unsupported NaNs, invalid units, insufficient strata, and test-fit
artifacts.

## 10. Validation gates

No categorical detection result is released until simulation tests cover:

- perfect generators and pointwise false-positive behavior;
- detectable shifts, variance changes, crossing CDFs, and tail contamination;
- balanced and severely unequal block sizes;
- thin strata and fewer-than-ten-block failure paths;
- localized copying, mode collapse, excessive diversity, and off-manifold data;
- monotone perturbation-severity response;
- global condition-mixture mismatch;
- xarray ordering, lazy arrays, NaNs, invalid units, and block overlap;
- deterministic regeneration of reports and figures.

## 11. Tropical-cyclone brightness-temperature case study

The demonstration evaluates `baseline_ztsnr` storm-centered infrared brightness
temperature fields using USA_WIND one-minute intensity labels and storm-disjoint
train, validation, and test partitions. Native conditions are TS and Categories
1–5. Category 5 remains visible despite its thin reference; a preregistered C4+
extreme-tail aggregate is supplemental and never replaces C4 or C5.

The seven registered scalar functionals cover core temperature, cold-cloud
fraction, radial gradient location, axisymmetry, deviation-angle variance, eye
contrast, and the cold BT tail. Radial-profile and radial-PSD adapters add
spatial-structure diagnostics. Training-fitted feature metrics, localized
copying, dependence closure, and sensitivity controls complete the evaluation.

The paper-facing evidence is:

- the class-aware reference-gap profile with degenerate-model anchors;
- a native-unit metric table;
- tail-focused physical distribution and spatial-structure figures;
- the complete class-by-metric appendix matrix;
- sensitivity, memorization, provenance, and calibration audit material.

## 12. Permitted claims and publication position

GAUGE may claim that it:

- supplies a portable, dependence-aware, reference-calibrated evaluation
  protocol for generated climate fields;
- makes heterogeneous metrics interpretable at a declared sample size while
  retaining native-unit evidence;
- integrates physical, conditional, diversity, memorization, admissibility, and
  sensitivity evidence in an extensible report;
- exposes failures hidden by global or unreferenced metric summaries;
- makes a rigorous evaluation easy to run and easy to present.

GAUGE must not claim that:

- reference-envelope overlap proves equality or realism;
- the reference gap is a universal physical severity scale;
- any built-in functional suite is universal across Earth-system applications;
- a single aggregate score can rank models safely;
- corruption controls or reference boundaries define application acceptability.

The strongest paper contribution is the protocol and reporting design, supported
by statistical validation and a nontrivial case study—not the performance of any
particular generator.

## 13. Remaining release gates

Before submission:

1. complete an independent statistical review of the reference engine,
   resampling intervals, simultaneous inference, and simulation coverage;
2. finish unit/integration tests and clean-environment package installation;
3. make the zero-configuration CLI/API produce the same paper bundle as the TC
   workflow;
4. archive permission-cleared data manifests, checksums, configurations, and
   reports;
5. regenerate every manuscript figure and table from archived reports;
6. complete a clean-room reproduction and usability test with target users.
