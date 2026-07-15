# GAUGE — Paper blueprint and implementation specification

**GAUGE** — *Guidelines for the Assessment and Uniform reporting of Generative Earth-system models.* A gauge is a measuring instrument with a declared resolution; the framework makes the metaphor literal by reporting every discrepancy together with its sampling reference, uncertainty, application tolerance (when one is meaningful), and sensitivity at the declared sample size.

This document is both the paper blueprint and the engineering spec: every algorithm is specified to the level of defaults, sampling rules, and pseudocode, so that the reference implementation can be written from this document alone.

Working title: **"Realism is not utility: a two-track protocol for evaluating generative models in climate science."**
Alternates: "Good enough for what? Evaluating generative Earth-system models against floors, tolerances, and downstream value." / "GAUGE: an evaluation protocol for generative models of geophysical fields."

---

## 0. Thesis and scope

**Thesis.** Evaluating a generative model of geophysical fields requires two tracks, and neither track is meaningful without an explicit reference:

- **Track 1 — Intrinsic realism.** A battery of metrics spanning physical fidelity, diversity/coverage, and memorization. Every result carries (i) a raw effect and uncertainty, preferably in physical units; (ii) a **real-reference envelope** produced by a dependence-matched simulation of a perfect generator at the declared sample size; and, where defensible, (iii) a predeclared application tolerance. The reference envelope answers whether a departure was detected at this `N`; a direct confidence bound relative to the tolerance answers whether the discrepancy is small enough for the application. These are separate conclusions.
- **Track 2 — Extrinsic utility.** A downstream task in which generated samples are consumed, evaluated against **information-matched baselines** (the cheapest procedures that inject the same reweighting and invariance information without a learned generator) and restored-real positive controls. Generative added value is the paired increment above a baseline with the same sample counts and transforms. Restored real data establishes empirical headroom; it is not assumed to be a mathematical upper bound.

Track 1 without Track 2 can be gamed by models that are distributionally faithful but informationally sterile; Track 2 without Track 1 cannot localize *why* a model helps or fails. The tropical-cyclone case study is designed to demonstrate this possible dissociation without presupposing its result. The working survey corpus (`n = 69`; §6) motivates the protocol, but all headline denominators remain provisional until the systematic-search reconciliation and human verification are complete.

**Scope condition.** Track 2 applies when the model is a *data provider* (augmentation, emulation, scenario generation, synthetic archives). When the model *is* the forecast, extrinsic evaluation is standard probabilistic verification (CRPS, rank histograms, spread–error) and the protocol defers to that tradition.

**Supported scope in v1.** GAUGE is a portable protocol with adapters, not a universal metric collection. The reference implementation supports independent generated samples of 2-D single- or multichannel gridded fields. Users may register domain functionals, embeddings, and metrics. Arbitrary spatiotemporal trajectories, spherical-vector geometry, coupled multivariate conservation, and forecast verification require domain adapters or established verification packages and are not claimed as automatic v1 capabilities.

**Non-goals.** GAUGE is not a task benchmark (no fixed dataset or leaderboard), does not define universal acceptability thresholds (tolerances are declared and justified by the user), does not treat a non-rejection as proof of equality, and does not attempt to define perceptual quality.

**The one-sentence promise to the reader.** GAUGE tells the reader what departure was measured, whether that departure was detected relative to real-data sampling variability at the declared `N`, whether it was demonstrated to lie within a predeclared application tolerance, and whether generated data add downstream value beyond information-matched baselines.

---

## 1. Data model and notation

All arrays are `float32` unless stated. Every dataset carries block structure.

| Object | Type / shape | Meaning |
|---|---|---|
| `X_train` | `(N_tr, H, W)` (or channels-first variants) | Training fields |
| `X_test` | `(N_te, H, W)` | Held-out real test fields |
| `X_gen` | `(N_g, H, W)` | Generated fields |
| `blocks_train`, `blocks_test` | `(N,)` int/str | Dependence-block id per frame (storm id, event id, year). **Mandatory.** |
| `y_train`, `y_test`, `y_gen` | `(N,)` optional | Condition labels (e.g., intensity category) |
| `phi_registry` | ordered dict | Named functionals `f: field → scalar`, each with `units`, `orientation` |
| `tolerances` | dict | `δ_app` per functional, native units, declared **before** evaluation |
| `estimand` | enum | `block_balanced` (default) or `frame_weighted`; fixes the target distribution |
| `config` | dict | `K`, `seed`, `N_eval`, estimand, feature-space spec, kernel spec; hashed into every report |

**Split discipline (validity precondition).** Train/test separation must be at block level: no block contributes frames to both sides. Random frame-level splits on autocorrelated archives are invalid and the implementation must refuse them (`blocks_train ∩ blocks_test = ∅` is asserted). Recommended additional check: temporal-era sensitivity (train on earlier years, test on later) reported in an appendix.

**Estimand discipline.** The unit being represented must be declared before evaluation. Under the default `block_balanced` estimand, a draw selects a block uniformly and then a frame uniformly within that block; every storm/event/year therefore receives equal total weight. Under `frame_weighted`, a frame is selected uniformly and longer blocks receive proportionally more weight. Pseudo-generated and reference-real draws use the same rule and the same `N_eval`. Results from different estimands answer different scientific questions and are never silently combined.

**Feature space** (used by MMD², coverage/density, diversity, memorization): by default, the standardized functional matrix Φ concatenated with the leading `n_eof = 20` EOF/PCA coefficients. Standardization statistics and EOF basis are **fit on `X_train` only** and frozen. A user-supplied, training-fitted embedding is also valid and must record its provenance. Kernel bandwidth for MMD² is selected once by a declared training-only rule; test-set tuning is forbidden.

---

## 2. Track 1 — intrinsic evaluation

### 2.1 Functional suite Φ and metric registry

**Φ declaration discipline.** 5–10 scalar functionals, chosen *before* evaluation, spanning **amplitude**, **spatial structure**, and **extremes**, each with units and a one-line physical justification. At least one functional must be tail-referenced.

Default Φ for the tropical-cyclone brightness-temperature case study:

| id | Functional | Units | Aspect |
|---|---|---|---|
| `min_core_bt` | Minimum BT within r < 100 km of center | K | amplitude / intensity |
| `cold_frac_220` | Fraction of pixels with BT < 220 K | % | amplitude / convective area |
| `r_maxgrad` | Radius of max azimuthal-mean BT gradient | km | structure |
| `axisym` | RMS azimuthal deviation from azimuthal mean | K | structure / organization |
| `dav` | Deviation-angle variance | deg² | structure / organization |
| `eye_contrast` | Eye − eyewall azimuthal-mean BT contrast | K | structure |
| `p01_bt` | 1st percentile of pixel BT | K | tail |

**Metric registry** (each entry: `id`, `type`, `estimator`, `orientation transform` so that *larger = worse* always holds after transformation):

1. **Per-functional W1** (`type=functional_w1`, one per Φ entry). Exact 1-D Wasserstein-1 between generated and real distributions of the functional. Equal-size samples: `mean(|sort(x) − sort(y)|)`; unequal: quantile-function integral (`scipy.stats.wasserstein_distance`). Units: the functional's own. This is the **currency metric**: under a pure location shift δ, W1 = δ exactly (the *shift identity*), so "W1 = 0.5 K on `min_core_bt`" reads as "differs by as much as a 0.5 K uniform shift would." Each W1 carries a **direction word** derived from `sign(median(gen) − median(real))` mapped to a per-functional vocabulary ("too warm/too cold", "too axisymmetric", …).
2. **Radial profile MAE** (`type=curve`). Azimuthal-mean BT profile per sample on fixed radial bins (default 32 bins to 500 km); scalar = MAE between sample-mean profiles; display = both mean curves + a dependence-matched real-reference envelope.
3. **Radial PSD** (`type=curve`). Per sample: 2-D FFT, |·|², radial wavenumber binning (log-spaced, 24 bins); scalar = MSE between mean log₁₀-PSD curves; display = curves + envelope.
4. **Feature MMD²** (`type=set`). Unbiased estimator, Gaussian kernel, feature space and bandwidth per §1. Orientation: raw value.
5. **Coverage** and **Density** (`type=set`; Naeem et al. 2020), k = 5 NN in feature space. Orientation transform: `max(0, 1 − value)` for coverage; `|1 − value|` for density.
6. **Diversity departure** (`type=set`). Absolute departure of the generated/real mean pairwise feature-distance ratio from 1; direction is reported separately as insufficient or excessive diversity, so overdispersion is not rewarded.
7. **Memorization** (`type=three_sample`, §2.7). Not reference-position standardized; reported as its own test.
8. **Admissibility** (`type=rates`, §2.7). Hard violation rate (must be 0) and soft envelope exceedance.
9. **Dependence** (`type=matrix`, §2.7). Spearman correlation matrix of Φ, per-entry gaps.

### 2.2 The dependence-matched real-reference engine

The reference must simulate a perfect generator while preserving the declared estimand. Comparing a block-balanced pseudo-generator with all frames of the reference half changes the estimand when block sizes differ and is invalid.

**Algorithm (per metric, per stratum):**

```
inputs: X_test, blocks_test, X_gen, N_eval, K (default 200), estimand, seed
for k in 1..K:
    split eligible test blocks uniformly into disjoint halves A_k, B_k
    pseudo_k = draw N_eval samples from A_k using estimand
    real_k   = draw N_eval samples from B_k using the same estimand
    gen_k    = draw N_eval generated samples without replacement
    reference_k = metric(pseudo_k, real_k)
    model_k     = metric(gen_k, real_k)  # paired through the same real_k
outputs:
    real-reference envelope {reference_k}
    paired model-reference distribution {model_k, reference_k}
    stored block partitions and draw provenance
```

Hard error if either half has fewer than 10 eligible blocks; warning below 30. `N_eval` may not exceed the generated pool. A split-half Monte Carlo stability diagnostic compares reference boundaries from the first and second halves of the registered draws; instability requires increasing `K` before a final run, never changing the conclusion rule. Conditional strata are formed through whole-block eligibility and splitting so that one storm can never appear on both sides.

### 2.3 Two conclusions, not one verdict

**Detection conclusion.** A model is `DEPARTURE_DETECTED` when its statistic lies beyond the calibrated one-sided real-reference boundary at the declared `N`; otherwise it is `NO_DEPARTURE_DETECTED`. The latter means only that this evaluation did not detect a departure. It is not evidence that the population distributions are equal. Joint claims require an explicit preregistered family of exact `stratum:metric` keys; the default empty family makes no joint claim. Version 0.1 uses cell-wise Bonferroni reference boundaries, which remain valid when conditional strata do not admit a common block partition. Their finite-Monte-Carlo order statistic requires `K ≥ family_size / α − 1`; otherwise the joint conclusion is withheld. A less conservative max-statistic method may be registered only when every cell genuinely shares resampling units and draws.

**Practical conclusion for functionals.** Classical bootstrap intervals for empirical Wasserstein distances can fail, especially near equality, and are not the inferential default. Under the default block-balanced estimand, select one frame uniformly from each independent real block and construct a finite-sample W1 interval from simultaneous DKW quantile bands (Manole, Balakrishnan & Wasserman, 2022). Untrimmed W1 requires a declared finite functional support; otherwise report a trimmed W1 interval or withhold equivalence. Clustered bootstrap intervals remain explicitly labeled diagnostics until their coverage is validated for the declared data-generating regime. With predeclared tolerance `δ_app`:

1. `WITHIN_TOLERANCE` if the one-sided upper confidence bound is below `δ_app`;
2. `EXCEEDS_TOLERANCE` if the lower confidence bound is above `δ_app`;
3. `UNDERPOWERED` if the interval overlaps `δ_app` and the predeclared sensitivity check cannot resolve a shift of that size;
4. `INCONCLUSIVE` otherwise.

Non-functional metrics receive no tolerance verdict unless the user supplies a scientifically meaningful discrepancy scale and a validated inference procedure for it.

### 2.4 Standardized reference position and raw effect

For diagnostic plotting only:

```
reference_position = (model_median - reference_q50) /
                     (reference_q(1-alpha) - reference_q50)
```

Position 1 marks the pointwise reference boundary. This dimensionless value describes detectability at this `N`, not physical severity. It is never averaged across metrics, compared across sample sizes, or used as an equivalence test. Every position is displayed beside the raw effect and confidence interval; functional W1 remains in the functional's physical units.

### 2.5 Shift sensitivity

For each `functional_w1`, report the smallest uniform shift detected with 80% probability by the reference procedure. This quantity is named `shift_resolution` because the result is perturbation-specific; it does not validate sensitivity to variance, tail, or dependence errors and does not turn non-detection into equivalence. Other metrics use predeclared perturbation families with severity sweeps.

### 2.6 Reference and corruption lanes

Reference lanes are validation experiments, not model results:

| Lane | Construction | Role |
|---|---|---|
| `REAL_REFERENCE` | Cross-fitted pseudo-generator from disjoint test blocks | Checks pointwise and simultaneous false-positive control; conservatism is measured and reported |
| `COPYIST` | Uniform training-sample replay | Positive control for localized memorization; distributional behavior is reported, not assumed |
| `CLIM` | Training composite mean plus declared training-fit variability | Low-order-statistic control |
| `PHASE_SHUFFLE` | Training samples with Fourier phase randomized | Spectrum-preserving organization control |
| `CROSS_CLASS` | Training frames from the wrong condition | Conditional-fidelity control |
| `BLUR` | Training frames blurred over a predeclared severity grid | Spectral/spatial sensitivity control |

Each metric declares which perturbation it should detect and the scientifically interpretable severity grid. Sensitivity is validated by the response curve and detection probability; there is no universal `reference_position > 3` rule.

### 2.7 Q3/Q4 specifics

**Memorization — localized, block-aware three-sample test.** Fit the feature space and spatial partition on training data only. Within each occupied feature-space region, compare generated-to-training nearest-neighbor distances with held-out-real-to-training distances. Uncertainty resamples generated queries and whole held-out blocks; region-wise results use simultaneous or multiplicity-adjusted intervals. Generated samples significantly closer in any supported region imply localized copying; significantly farther imply a local off-manifold failure. A global summary is secondary because copying and underfitting in different regions can cancel. `COPYIST` must trigger the test.

**Admissibility.** Distinguish physical laws from preprocessing or sensor support. Report hard-constraint violation rates directly. Report soft training-envelope exceedance beside held-out-real exceedance with a dependence-matched reference. A bound enforced identically by the output activation is labeled a construction guarantee, not empirical evidence of learned physics.

**Dependence closure.** Report generated and real Spearman matrices of Φ and entry-wise differences with block-aware intervals. Sliced-W1 or another multivariate diagnostic may be registered as a complementary summary but never replaces the interpretable entries.

### 2.8 Conditional evaluation

When labels exist, metrics are computed globally and for predeclared strata. A stratum with fewer than 10 eligible blocks per half is reported `UNDERPOWERED`; scientifically justified pooled strata must be declared before results are inspected. Conditional ensemble calibration defers to established probabilistic verification where applicable.

### 2.9 Track-1 displays

The primary one-page card groups rows by physical fidelity, conditional fidelity, diversity/memorization, and admissibility. Functional rows show W1 and its interval in native units, the real-reference envelope, the tolerance, the detection conclusion, and the practical conclusion. A secondary profile uses standardized reference position on a symlog axis, prints `N_eval`, block counts, estimand, `K`, and seed, and explicitly labels the axis "detectability, not severity." No aggregate model score is produced.

---

## 3. Track 2 — extrinsic (downstream) evaluation

### 3.1 Principle: information-matched baselines

A generative model trained on the same data is a data *processor*, not a data *source*: its samples carry no information about the true distribution beyond training data + inductive biases. Downstream gains from augmentation therefore decompose into three rungs: **(1) reweighting** (oversampling minority classes reshapes empirical risk — obtainable by copying), **(2) invariance injection** (rotations/shifts encode physical symmetry priors — obtainable by cheap transforms), **(3) learned distributional structure** — the only rung a generative model can uniquely claim. Consequently, **the null hypothesis for generative augmentation is not "no augmentation"; it is the cheapest baseline occupying rungs 1–2.** Generative added value is *defined* as the increment above copy + invariances. Comparisons against no-augmentation conflate rungs and systematically overstate generative value.

### 3.2 The ladder (arms)

The augmented source arms A2–A7 use the same declared target counts. A0 retains the restricted empirical counts and A1 represents their contribution through loss weights rather than duplicating rows. All arms train the identical downstream architecture with identical hyperparameters, apart from those predeclared sampling/weighting differences. Default: `S = 5` seeds per arm (10 if cheap); seed controls initialization, batching, and augmentation sampling.

| Arm | Construction |
|---|---|
| `A0 no-aug` | Empirical training distribution as-is |
| `A1 class-weight` | No new samples; inverse-frequency class weights in the loss (isolates rung 1 without data duplication) |
| `A2 copy` | Minority frames duplicated to target counts (pure rung 1) |
| `A3 copy+geo` | A2, then geometric transforms on the copies — **declared transform set**: rotations {±45°, ±90°, ±135°, 180°}, integer translations {±1, ±2} px, sampled uniformly (rungs 1+2) |
| `A4 gen` | Minority counts topped up with generative samples, count-matched to A2/A3 (rung 3 candidate) |
| `A5 gen+geo` | A4 plus the A3 transform set on generated samples (tests complementarity) |
| `A6 restored-real` | Withheld real blocks restored, no geometry |
| `A7 restored-real+geo` | A6 plus the identical declared transform policy |

**Primary contrast:** `A5 − A3` = generated+geometry versus copy+geometry. **Secondary source contrast:** `A4 − A2` = generated versus copy without geometry. `A4 − A3` is forbidden because it changes both sample source and transform policy. **Reproduction contrast:** `A4 − A0`, shown only to demonstrate how a weak baseline can alter the apparent gain.

### 3.3 Restored-real positive controls (holdout-restore design)

The restored-real arms ask whether additional real information helps under the same sample-count and transform policy. They establish empirical headroom, not a theoretical ceiling: synthetic data can sometimes regularize or target a task more effectively than an equal number of restored real samples.

```
1. Withhold a fraction ρ = 0.5 of minority-class blocks (storms, not frames)
   from ALL arms' training data → the "restricted" training set.
2. Train the generator used by A4/A5 on the restricted set only.
3. All arms A0–A5 run on the restricted set; augmentation targets are
   count-matched to the frame count of the withheld blocks.
4. A6 restores real blocks without transforms; A7 restores them with the
   identical geometry policy used by A3/A5.
5. Report A6 − A2 and A7 − A3 as headroom controls alongside A4 − A2 and
   A5 − A3 as source-matched generative contrasts.
```

Withholding at block level is mandatory: selecting a storm removes every frame from that storm, even when it spans multiple categories. The holdout is selected before generator training. Using a generator trained on restored blocks invalidates the contrast. The case study uses three predeclared holdouts; results are reported per holdout and hierarchically.

### 3.4 Statistics, metrics, and equivalence

- **Predeclared endpoints:** the TC case study uses continuous Vmax regression as primary and category classification as secondary. Neither is promoted or replaced after inspecting restored-real results. The regression metric and `δ_task` are frozen before the three restricted generators are trained.
- **Uncertainty:** pair arms by holdout and optimizer seed, and resample test storms rather than individual frames. Report the three holdouts separately and a hierarchical summary; seeds alone do not represent data uncertainty.
- **Equivalence:** orient every contrast so positive means downstream improvement. A direct paired confidence interval wholly within `(−δ_task, +δ_task)` supports task equivalence; an interval wholly above zero supports superiority; otherwise the result is inconclusive. Multiplicity is controlled for any family of co-primary contrasts.
- **Display:** arms on y, metric on x, per-seed scatter and interval, with restored-real arms visually marked as positive controls. The headline panel is the matched `A5−A3` contrast; `A4−A2` is the source-only contrast, `A7−A3` shows empirical headroom, and `A4−A0` is labeled a weak-baseline reproduction comparison.

---

## 4. Box 1 — the protocol (verbatim, one boxed page in the paper)

> **Step 0 — Declare, before evaluating.** (a) Block-level train/test separation (state the block unit; frame-level splits on autocorrelated data are invalid). (b) A functional suite Φ of 5–10 scalar physical quantities spanning amplitude, structure, and extremes, with units; at least one tail-referenced. (c) Per functional, a tolerance δ_app with a one-sentence justification. (d) Evaluation sample size N = the number of samples your application will consume. (e) The downstream task, its metric, its minimal relevant improvement δ_task, and the augmentation ladder (if the model is a data provider). (f) K (default 200) and the seed.
>
> **Step 1 — Declare the estimand and build the real-reference envelope.** Choose block-balanced or frame-weighted sampling. Simulate a perfect generator and its real comparator from disjoint block halves using the same estimand, sample size, and partitions used for the model comparison.
>
> **Step 2 — Separate detection from practical relevance.** Report `DEPARTURE_DETECTED` or `NO_DEPARTURE_DETECTED` relative to the reference envelope. For functional W1, compare the validated dependence-aware confidence bound directly with `δ_app` to report `WITHIN_TOLERANCE`, `EXCEEDS_TOLERANCE`, `INCONCLUSIVE`, or `UNDERPOWERED`. In v0.1 the formal block-balanced bound uses one frame per independent block and simultaneous DKW bands; the whole-block bootstrap is diagnostic. Non-detection is not equivalence.
>
> **Step 3 — Marginal realism (Q1).** Per-functional W1 in native units; radial-profile and radial-PSD curves with real-reference envelopes; the tail functional explicitly.
>
> **Step 4 — Conditional fidelity (Q2; skip and say so if unconditional).** Predeclared per-category analyses with block-stratified references; report thin strata as UNDERPOWERED and use only predeclared pooled strata; defer to standard verification for high-dimensional forecast conditioning.
>
> **Step 5 — Admissibility (Q3).** Hard-violation rate (must be zero, reported as such); soft envelope exceedance with its real-reference band.
>
> **Step 6 — Diversity and memorization (Q4).** Coverage, density, and diversity in a declared training-fit feature space; a localized block-aware three-sample copying test with a copyist positive control.
>
> **Step 7 — Validate sensitivity.** Evaluate cross-fitted real-reference and predeclared corruption lanes at the same N. Verify, with Monte Carlo uncertainty, that perfect-generator false positives do not exceed the declared α; report any conservatism rather than claiming exact calibration. Validate each metric with an appropriate perturbation severity curve; do not apply a universal reference-position threshold.
>
> **Step 8 — Downstream value (when the model provides data).** Run the count-matched eight-arm ladder: no augmentation, class weights, copy, copy+invariances, generated, generated+invariances, restored real, restored real+invariances. Select block holdouts before generator training. Headline generated+invariances versus copy+invariances; treat restored real as empirical headroom, not a bound.
>
> **Step 9 — Report.** The one-page evaluation card, secondary standardized profile, ladder figure, and machine-readable report. Every number carries its sampling reference, uncertainty, estimand, N, and permitted interpretation. No aggregate model score is produced.

---

## 5. Reference implementation spec (`gauge-climate`, pip-installable)

### 5.1 Package layout

```
gauge_climate/
  data.py        # Dataset container: fields, blocks, labels; split validators
  types.py       # Dataset-adjacent specs, config, status and report-cell types
  adapters/      # domain functionals and curve metrics; TC-BT in v0.1
  features.py    # standardization + EOF basis (train-fit, frozen); kernels
  metrics.py     # w1, curve metrics, mmd2, coverage/density, diversity,
                 # spearman matrix, admissibility rates
  reference.py   # dependence-matched real-reference engine (§2.2)
  inference.py   # bounded-W1 bounds, clustered diagnostics, simultaneous bands,
                 # and two-layer conclusions
  lanes.py       # perturbations and predeclared response sweeps (§2.5–2.6)
  memorization.py# block-aware three-sample test (§2.7)
  downstream.py  # ladder orchestration, holdout-restore, equivalence stats (§3)
  report.py      # versioned card JSON, HTML, and diagnostic profile figure
  xarray_adapter.py # optional dimension-aware lazy-array adapter
  cli.py         # `gauge-climate validate|run|report`
```

### 5.2 Core API

```python
import gauge_climate

ds_real  = gauge_climate.Dataset(X_test, blocks=storm_ids_test, labels=cats_test)
ds_train = gauge_climate.Dataset(X_train, blocks=storm_ids_train, labels=cats_train)
ds_gen   = gauge_climate.Dataset(X_gen, labels=cats_gen)
functionals = gauge_climate.adapters.tc_bt.functionals()

feature_space = gauge_climate.features.TrainFittedFeatureSpace.fit(
    X_train, functionals, n_eof=20, fit_source="train"
)
feature_metrics = feature_space.registered_metrics(X_train)
copying = gauge_climate.memorization.localized_copying_test(
    train_features=feature_space.transform(X_train),
    real_features=feature_space.transform(X_test),
    generated_features=feature_space.transform(X_gen),
    real_blocks=storm_ids_test,
)

report = gauge_climate.evaluate(
    real=ds_real, generated=ds_gen, train=ds_train,
    functionals=functionals,
    metrics=feature_metrics,
    tolerances=predeclared_tolerances,   # frozen native-unit values
    config=gauge_climate.EvaluationConfig(
        n_eval=None, n_resamples=200, seed=1789, estimand="block_balanced"
    ),
    memorization_result=copying,
    sensitivity_results=precomputed_sensitivity_lanes,
    conditional=True,
)
report.figure_reference_profile(path="fig_reference_position.pdf")
report.to_json("evaluation_card.json")

ladder = gauge_climate.downstream.Ladder(
    task=my_train_eval_fn,  # (ArmBatch, seed, holdout_id) -> metric or per-block metric
    task_metadata=frozen_task_and_optimizer_config,
    train=ds_train,
    holdouts=frozen_holdouts,
    generated_by_holdout=retrained_generated_pools,
    generator_training_blocks_by_holdout=restricted_training_storms,
    transform=gauge_climate.downstream.geo_default,
    matching_strata=frozen_wind_bin_function,
    matching_metadata={"variable": "Vmax", "edges_kt": frozen_edges},
    config=gauge_climate.downstream.LadderConfig(
        seeds=(101, 102, 103, 104, 105)
    ),
)
ladder_report = ladder.run()
primary = ladder_report.contrast(
    metric=primary_endpoint,
    treatment=gauge_climate.downstream.Arm.A5_GENERATED_GEO,
    reference=gauge_climate.downstream.Arm.A3_COPY_GEO,
    delta_task=frozen_delta_task,
    higher_is_better=False,
)
```

`my_train_eval_fn` is user-supplied (any framework); `gauge_climate` orchestrates arms, count-matching, block-level holdout, seeds, and statistics — it never touches the model internals.
The built-in geometry callback persists every rotation/translation and an output
hash. A custom transform may return `(fields, trace_mapping)` to provide the
same exact provenance; array-only callbacks retain their seed, selected rows,
frozen callback metadata, and output hash but are marked as lacking explicit
transform parameters.

### 5.3 Evaluation-card JSON (schema sketch)

```json
{
  "schema": "gauge_climate.evaluation.v1",
  "metadata": {"gauge_version": "0.1.0", "config_hash": "…",
               "estimand": "block_balanced", "n_eval": 1000},
  "cells": [{
     "metric": "w1:min_core_bt", "stratum": "cat5",
     "pillar": "conditional_fidelity", "units": "K",
     "estimate": "<raw>", "ci_low": "<bound>", "ci_high": "<bound>",
     "reference_q50": "<q50>", "reference_detection_boundary": "<q(1-alpha)>",
     "reference_position": "<diagnostic>", "shift_resolution": "<native units>",
     "detection_status": "<two-layer detection value>",
     "practical_status": "<tolerance value>",
     "sensitivity_status": "<control value>", "direction": "<physical wording>"
  }],
  "memorization": {"status": "NO_LOCAL_DEPARTURE_DETECTED", "regions": []}
}
```

Track 2 is serialized separately as `gauge_climate.downstream.v1`, including
per-holdout/seed/arm records, per-test-block metrics, selected row indices,
matching strata, transform metadata, and frozen task metadata.

### 5.4 Implementation notes and gotchas

- Every resampling stage uses a deterministically derived seed; block and row partitions are persisted for exact reruns.
- Functional matrices Φ computed once per dataset and cached; metrics operate on Φ, features, or raw fields per registry `type`.
- MMD² unbiased estimator excludes diagonal terms; subsample to ≤ 5,000 per side per partition for O(n²) tractability, documented.
- Coverage/density need the *real* manifold radii recomputed per partition (they depend on the real half used).
- Benchmark runtime and peak memory on the archived case-study configuration; do not infer computational cost from exploratory runs.
- Refuse to run if `blocks` are missing or train/test blocks intersect. Conditional strata below the block minimum are emitted as `NOT_EVALUATED` / `UNDERPOWERED`, never silently pooled or forced through.

---

## 6. The provisional literature survey (`n = 69`): evidence and hardening

**Status.** The supplied workbook contains 69 coded papers (2020–2026), one row per paper, and four additional candidate rows. The earlier planning count of five outstanding candidates is therefore not supported by the current artifact and must be reconciled against the frozen search log. Corpus assembly was two-phase: 41 papers from the original collection plus 28 added purposively to broaden model-class coverage (GAN/VAE/flow alongside diffusion). Coding was a single-coder pass with LLM-assisted adversarial verification on the second batch—working-grade, not publication-grade. `survey/export_survey.py` now exports canonical rows and recomputes all criterion counts without reading the workbook's Summary sheet; its checksum manifest and `headline_use_permitted: false` flag make the current evidence state explicit.

**Headline results** (strict = yes only; lenient = yes + partial; report both everywhere — the current Summary sheet mixes them silently):

| Criterion | Strict | Lenient |
|---|---|---|
| Q4b memorization / NN-to-training check | **1 (1.4%)** | 1 (1.4%) |
| Q5a discriminative two-sample test (C2ST) | 0 | 1 (1.4%) |
| Q5b any downstream-utility evaluation | 6 (9%) | 12 (17%) |
| U2 any real-vs-real reference | 14 (20%) | 16 (23%) |
| V1 any metric-validity control | 10 (14%) | 25 (36%) |
| U1 UQ on the metrics themselves | 16 (23%) | 23 (33%) |
| Q4a diversity/coverage quantified | 14 (20%) | 30 (43%) |
| Q1a quantitative distributional comparison | 31 (45%) | 42 (61%) |
| Q1b visual-only distribution claims | 15 (22%) | 21 (30%) |
| Q1c spatial spectra quantified | 18 (26%) | 18 (26%); 5 visual-overlay-only rows are reported separately |
| B1 non-ML baseline | 44 (64%) | 50 (72%) |
| Held-out test set present | 66 (96%) | — |
| Leakage-prone splits | 5 (7%) | 16 (23%, incl. unclear) |

Any aggregate crossing multiple criteria must be generated by the survey scripts before it is used. Earlier manually transcribed cross-criterion totals are not part of the frozen evidence base.

**Working hypothesis signals—not manuscript conclusions.** The current rows suggest that diversity and memorization checks are uncommon; real-reference ideas are present often enough that GAUGE must be positioned as a formalization; and matched-information downstream controls may be absent. Venue, temporal-trend, leakage, and downstream-design comparisons require regenerated cross-tabs plus full-text verification. No hypothesis is labeled confirmed or disconfirmed until the frozen corpus, independent agreement, and load-bearing verification gates pass.

**The repositioning this forces (C2).** The U2 notes surface direct prior art for reference-floor ideas: a 2023 kilometer-scale GAN-emulation study comparing independent real batches, DiffESM's two-real-realizations reference, an extratropical-cyclone generation study with PSD and energy-distance natural-variability references, and TemperatureGAN's statistic-matched reference generator. These papers require full-text human verification before they carry the related-work argument. In the current export, 14/69 are strict `yes` and 16/69 are `yes + partial`; the denominator and coding remain provisional. The intended claim is therefore not that GAUGE invented real-reference comparison, but that it formalizes and packages an emerging practice while adding explicit estimands, dependence matching, detection semantics, direct tolerance inference, sensitivity validation, and localized copying tests.

**Provisional abstract evidence sentence (not usable until hardening).** The current workbook supports a sentence of this form, but every count and denominator must be regenerated after the frozen systematic search, eligibility reconciliation, independent double-coding, and full verification of the load-bearing fields. The manuscript must not call the present hybrid corpus a systematic survey.

**Hardening checklist (replaces the pilot gate, which the executed survey has mooted; this is the survey-side critical path to publication grade):**
1. **Human double-coding** of the deterministic 20% (`n=14`) blinded subsample by an independent coder; report raw agreement and Cohen's κ, plus Gwet's AC1 for rare criteria. Tighten rules and re-code fields with inadequate agreement. Human verification of the four load-bearing fields (Q4b, U2, Q5b, V1) on the **full** corpus, not only the second batch.
2. **Corpus documentation.** Either document the two-phase assembly (41 + 28 purposive) as a stratified design with the retro-specified search protocol and a PRISMA-lite flow diagram, or re-run the systematic query and reconcile. Disclose the LLM-assisted screening/verification and its human-verification policy explicitly.
3. **Reconcile and code the outstanding candidates.** The supplied candidate sheet has four rows; resolve the discrepancy with the earlier count of five from the frozen search log rather than assuming either number.
4. **Add the downstream-baseline-strength column** ∈ {none, no-aug, weak, matched-information, restored-real control}; it is the H7 evidence field.
5. **Eligibility re-check** on borderline rows (land-cover classification vs geophysical fields).
6. **Reporting convention:** strict and lenient counts side by side, everywhere.

**Outputs.** The repository now contains source-to-CSV export, a generated codebook, deterministic blinded 20% sample (`n=14`), strict/lenient criterion summaries, and file checksums. Dot matrix, coverage bars, screening-flow diagram, independently recoded agreement output, and every cross-criterion aggregate remain gated on the frozen protocol and verified corpus.

**Fairness clause.** Papers claiming augmentation gains against weak baselines are engaged by name in the Discussion only after the downstream-baseline-strength field is human-verified. GAUGE may show the conventional reproduction contrast (`A4−A0`) descriptively, but its inferential contrasts are the count-matched `A4−A2` and transform-matched `A5−A3`; design differences must be acknowledged before attributing a reported gain to reweighting or invariance.

---

## 7. Case study — tropical-cyclone brightness-temperature generation

**Data.** Storm-centered infrared BT archive; block unit = storm; labels = the six categories actually represented by the generator and prepared data (TS, Cat 1–5). TD is not claimed. Document storms and frames per side; temporal-era sensitivity split in the appendix. Report the count table (storms × frames × category × split) as Table 1. Predeclare Cat 5 as thin and a scientifically justified C4+ pooled analysis before the final run.

**Models.** Evaluate the existing baseline and physics-informed diffusion variants under the same frozen protocol, while designating one primary model before the final test run. Additional model families are optional and are de-scoped before compromising the preregistered comparisons. Reference lanes follow §2.6.

**Track 1 (must be rerun under the frozen protocol).** Evaluate per-functional W1, radial profile, radial PSD, feature MMD², coverage/density/diversity, localized memorization, admissibility, and Spearman closure globally and per predeclared category. Existing exploratory runs may motivate diagnostics but do not establish paper results because they use a different reference engine and test-fit feature construction.

**Track 2 (pending, preregistered design).** Primary task: Vmax regression; secondary task: intensity-category classification. Select three storm-level Cat 4–5 holdouts before training and remove every frame belonging to each selected storm. Train one restricted generator per holdout, generate at continuous wind conditions drawn from the permitted training distribution, and run A0–A7 with five paired downstream seeds. Freeze the primary metric and `δ_task` before training. Headline `A5−A3`; show `A7−A3` as restored-real headroom. Classification cannot replace regression after results are observed.

**Track 1 ↔ Track 2 linkage (write with care).** The tracks answer different questions—realism and value—and neither result implies the other. Any proposed link between a detected intrinsic departure and a downstream effect requires a dedicated intervention or mediation analysis; it must not be inferred post hoc from coincident case-study results.

**Headline figures.** F1: hardened survey dot-matrix. F2: one-page Track-1 evaluation card. F3: standardized reference-position profile with sensitivity lanes. F4: Track-2 ladder headlining `A5−A3`, with `A4−A2` as the source-only contrast and `A7−A3` as empirical headroom. F5: per-category status heatmap. Appendix: reference calibration, construction comparisons, and resolution-vs-N curves.

---

## 8. Paper skeleton

1. **Introduction.** The evaluation gap; the copyist thought experiment (aggregate distributional metrics can miss localized replay, and replay adds no information beyond oversampling); contributions: (C1) the two-track principle; (C2) dependence-matched real-reference envelopes separated from direct tolerance inference; (C3) the information-matched downstream design with restored-real controls; (C4) the hardened survey; (C5) the case study; (C6) the package.
2. **What to evaluate.** Four questions (marginal realism; conditional fidelity; admissibility; diversity/memorization); Φ-declaration discipline; W1 as currency (units, shift identity, direction words).
3. **Against what.** The declared estimand; dependence-matched real-reference envelopes; direct finite-sample population-discrepancy bounds; two-layer conclusions; standardized reference position with its prohibitions; shift sensitivity; multiplicity; fit-for-purpose N.
4. **Downstream value.** The three rungs; matched eight-arm ladder; leakage-free holdout-restore controls; direct equivalence with `δ_task`; scope condition.
5. **Current practice.** Frozen survey protocol (full → Appendix B); generated flow diagram, dot matrix, agreement, and verified results; the repositioned claim that GAUGE formalizes and extends emerging real-reference practice.
6. **Case study.** Preregistered setup; Track-1 card and diagnostic profile; Track-2 regression ladder with classification secondary; matched and reproduction contrasts.
7. **Discussion.** Delegation of acceptability to declared tolerances; what GAUGE does not do; engagement with contradicted augmentation claims (fairness clause); adoption path (versioned living standard; plug-in ambition toward community evaluation infrastructure); limitations (Φ-dependence; 1-D marginal emphasis with the Spearman/sliced-W1 closures; single-domain demonstration; reference position's sensitivity-not-severity semantics).
8. **Appendices.** A statistics; B survey; C package quickstart (one screen from `pip install` to figure); D case-study details; E reference-construction comparison.

**Positioning paragraphs (Intro §1.2).** Task benchmarks answer "who wins here"; GAUGE is a portable protocol for "what departure is detectable and small enough for this use." Statistical-distance guides supply distance estimators; GAUGE supplies estimands, dependence-aware references, application tolerances, sensitivity validation, reporting semantics, and matched downstream controls. Cite improvised real-reference practices generously as the emergent practice GAUGE formalizes. The ML evaluation literature supplies density/coverage, two-sample, and copying diagnostics; forecast verification remains the authority when the generator is itself a forecast.

---

## 9. Venue, sequencing, risks, timeline

**Venue.** Primary: **GMD** (methods for assessment, open review, and reproducible implementation). Reconsider a stretch venue only from the strength of the fully preregistered results, never by switching endpoints or analysis branches after inspecting them.

**Sequencing and authority.** arXiv/EGUsphere preprint → climate-ML workshop presentation for community contact → journal submission with feedback provenance. Recruit one co-author with forecast-verification credibility and/or a recognized climate-ML name; the ask is bounded (§3 review + framing), the offer is a finished draft, working package, and the survey CSV. The pre-committed survey protocol and released CSV carry the standards argument so the authors don't have to.

**Risk register.** (R1) Survey counts change after reconciliation → all current percentages remain provisional. (R2) Restored real shows little headroom → report task saturation without changing the preregistered primary endpoint. (R3) Three restricted generator fits are heterogeneous → report each holdout and narrow the general claim. (R4) Statistical simulations fail nominal calibration → withhold categorical conclusions and revise the method before release. (R5) Schedule pressure → retain the validated reference engine, direct tolerance inference, one primary model, matched ladder, hardened survey, and reproducible package; de-scope secondary models and optional extrapolations first.

**Timeline (~4 months of new work; survey coding already done shifts the critical path to Track 2 and hardening).**
- **W1–2:** Freeze Φ, tolerances, `δ_task`, transform set, and PSD preprocessing. Survey hardening kickoff: assign the blinded `n=14` sample to an independent coder; freeze and code downstream-baseline strength; reconcile the four workbook candidates against the frozen search log; eligibility re-check. Package: validate the reference engine, tolerance inference, diagnostic reference-position profile, and sensitivity lanes.
- **W3–6:** Complete toy validation of all eight downstream arms, then run the frozen downstream architecture across the first restricted-generator holdout. κ checkpoint and full-corpus verification of Q4b/U2/Q5b/V1 complete; PRISMA-lite flow drafted.
- **W7–10:** Run the remaining two restricted-generator holdouts and final Track-1 evaluation. Produce final survey figures and reference-construction/resolution-vs-N appendix experiments. Draft §§1–5.
- **W11–14:** Full draft; figures; internal review; co-author ask with draft attached; EGUsphere/arXiv preprint; workshop submission per cycle.
- **W15–16:** Revise on early feedback; GMD submission. De-scoping ladder consulted at the W6 and W10 checkpoints, not in a crisis.
