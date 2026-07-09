# climategen-eval

`climategen-eval` is a standalone evaluator for generative climate-field models.
It is intentionally decoupled from diffusion-training code: it consumes real
reference data and model-generated sample banks, and emits standardized metric
reports plus publication figures. The tropical-cyclone brightness-temperature
case study is shipped end-to-end; the framework itself is meant to be
domain-portable.

## The framework

For a single climate-image case study the evaluator measures how far each
generative model sits from the real data distribution on a panel of physically
motivated metrics. Results are reported on a single interpretable axis:

1. **A real-vs-real noise floor.** For each metric, a storm-cluster bootstrap
   compares two finite samples of real images. The resulting null distribution
   sets the *good* anchor (the finite-sample median `q50`) and the
   *noise-floor boundary* (`q95` for lower-is-better metrics, `q05` for
   higher-is-better metrics).
2. **A standardized gap.** For a model, `gap = max(0, model − q50) / (q95 − q50)`.
   `gap = 0` matches the real-vs-real median; `gap = 1` sits exactly at the
   upper noise-floor boundary; `gap > 1` means the model is detectably worse
   than the real-vs-real noise can explain.
3. **Storm-cluster bootstrap everywhere.** Tropical-cyclone images from the same
   IBTrACS track are not independent observations, so all bootstraps resample
   storms (clusters) with replacement and take every image from each drawn
   storm. The plain iid bootstrap is reserved for genuinely independent rows
   (e.g. generated samples).
4. **Corruption-based metric validation.** A metric only appears on the gap
   profile if a known degradation family (Gaussian blur, pixel shuffle,
   brightness shift, center jitter, mode collapse) drives it past its noise
   floor at the configured power threshold. Metrics that do not respond to any
   degradation are flagged and excluded.
5. **A spanning set of targeted-null baselines.** Five parameter-free
   reference procedures, each preserving a specific sufficient statistic of
   the class-conditional distribution and nothing beyond it:
   - **GAUSS-FULL** — per-class maximum-entropy Gaussian with empirical
     mean and full spatial covariance. The textbook information-theoretic
     null for "everything learnable from the first two moments." Preserves
     the average radial PSD exactly; fails on higher-than-Gaussian structure
     (non-Gaussian pixel tails, gradient-angle statistics, per-image eye).
     Sampled via the linear-combination identity
     `x = mu + R^T z / sqrt(N - 1)` so the (H*W)^2 covariance is never
     materialized.
   - **CROSS-CLASS** — real images drawn from the *other* intensity classes.
     Orthogonal to any unconditional null; tests label/intensity fidelity.
   - **PHASE-SHUFFLE** — Theiler-style FFT phase randomization under
     Hermitian symmetry. Per-image surrogate: preserves each image's PSD
     and pixel histogram, destroys coherent spatial phase. Orthogonal to
     GAUSS-FULL (ensemble-level) since it operates per-image.
   - **CLIM** (ablation) — per-class pixel-wise mean replicated to fill the
     bank. Degenerate case of GAUSS-FULL with covariance set to zero.
   - **GAUSS-PIX** (ablation) — per-class mean plus iid per-pixel Gaussian
     noise. Degenerate case of GAUSS-FULL with off-diagonal covariance set
     to zero.
   The three primary nulls (GAUSS-FULL / CROSS-CLASS / PHASE-SHUFFLE) span
   complementary failure modes; CLIM and GAUSS-PIX are retained as
   ablations of GAUSS-FULL that strip progressively more covariance
   information out. A generative model is required to sit, on every metric,
   strictly inside the gap of every null that the metric is sensitive to.

The final figure overlays models and baselines on the same standardized-gap
axis. The x-axis defaults to a symmetric-log scale with `linthresh = 1` so the
noise-floor band stays in linear resolution while baselines that are dozens or
hundreds of widths out do not squash the model region.

### Connection to operational verification

The standardized gap is structurally a **skill-score-like quantity** in the
sense of operational weather forecasting: it normalizes a model error by the
distance between a *reference* anchor (here the real-vs-real finite-sample
median, the best a generative model can achieve at this evaluation set size)
and a unit of *detectable difference* (here the upper half-width of the
central 90% real-vs-real null). The bad end of the same axis is anchored by
the baseline panel rather than by a single climatology reference. The
methodology follows the spirit of Murphy (1988, *Mon. Wea. Rev.* 116), the
WMO verification guidelines (WMO No. 485), and Jolliffe & Stephenson,
*Forecast Verification* (2nd ed., Wiley 2012). A `null_standardization
.interpretation` block in every standardization report makes this framing
machine-readable.

### Per-image diagnostic distributions

The per-image diagnostics (cold-cloud fraction, DAV, eye-contrast proxy) are
compared between the real and generated sets via the 1-D Wasserstein-1
distance between their empirical per-image distributions, not by the absolute
difference of their means. This captures variance, skewness, and tail
mismatches that mean-comparisons miss: a model with the right mean DAV but
wrong DAV variance does not pass.

## Case-study sources

From the original `tc-diffusion` workspace:

- Real data split: `data/splits/files_test.txt`
- Real packed data: `data/packed_tc_memmap/manifest.json`
- Metadata: `data/dataset_index_wind.json`
- `pinn_loss` generated bank: `outputs/pinn_loss/sample_banks/test/final_bank`
- `FINAL_BASELINE` generated bank: `outputs/FINAL_BASELINE/sample_banks/test/final_bank`

The `FINAL_BASELINE` bank is inferred from the `final_eval_1000` timestamp: the
later reranked bank was created on 2026-05-01, after `final_eval_1000` was
written on 2026-04-30, so it cannot be the source for that eval.

## Quick start

Prepare a local case-study data directory from the original repo:

```bash
python case_studies/tropical_cyclone_bt/prepare_case_study.py \
  --source-repo ../tc-diffusion \
  --output data/tropical_cyclone_bt \
  --copy-mode symlink
```

Compute the null-standardization report (defines the noise floor for every
metric and runs the corruption-based responsiveness gate):

```bash
python scripts/compute_tc_bt_standardization.py \
  --config configs/tc_bt_case_study.yaml \
  --generated-n-per-class 1000 \
  --real-reference-reps 200 \
  --output outputs/tropical_cyclone_bt/standardization/test_n1000/metric_standardization.json
```

Evaluate the bad-but-sensible baseline panel against the standardization
report. This also emits a `baseline_panel.json` summarizing which metrics
each baseline activates (gap > 1) — the file the model evaluator and plotter
use as the metric-inclusion gate:

```bash
python scripts/evaluate_tc_bt_baselines.py \
  --config configs/tc_bt_case_study.yaml \
  --n-per-class 1000 \
  --standardization-report \
    outputs/tropical_cyclone_bt/standardization/test_n1000/metric_standardization.json \
  --output-root outputs/tropical_cyclone_bt/baselines
```

Evaluate the diffusion sample banks, gating metric inclusion on the baseline
panel just produced:

```bash
python scripts/evaluate_tc_bt_sample_banks.py \
  --config configs/tc_bt_case_study.yaml \
  --model FINAL_BASELINE=data/tropical_cyclone_bt/generated/FINAL_BASELINE/final_bank \
  --model pinn_loss=data/tropical_cyclone_bt/generated/pinn_loss/final_bank \
  --n-per-class 1000 \
  --standardization-report \
    outputs/tropical_cyclone_bt/standardization/test_n1000/metric_standardization.json \
  --baseline-responsiveness \
    outputs/tropical_cyclone_bt/baselines/baseline_panel.json \
  --output-root outputs/tropical_cyclone_bt/eval
```

Render the final standardized-gap figure with models and baselines on the same
axis:

```bash
python scripts/plot_model_metric_summary.py \
  --input FINAL_BASELINE=outputs/tropical_cyclone_bt/eval/FINAL_BASELINE/metrics.json \
  --input pinn_loss=outputs/tropical_cyclone_bt/eval/pinn_loss/metrics.json \
  --baseline GAUSS-FULL=outputs/tropical_cyclone_bt/baselines/gauss_full/metrics.json \
  --baseline CROSS-CLASS=outputs/tropical_cyclone_bt/baselines/cross_class/metrics.json \
  --baseline PHASE-SHUFFLE=outputs/tropical_cyclone_bt/baselines/phase_shuffle/metrics.json \
  --baseline CLIM=outputs/tropical_cyclone_bt/baselines/clim/metrics.json \
  --baseline GAUSS-PIX=outputs/tropical_cyclone_bt/baselines/gauss_pix/metrics.json \
  --standardization-report \
    outputs/tropical_cyclone_bt/standardization/test_n1000/metric_standardization.json \
  --baseline-responsiveness \
    outputs/tropical_cyclone_bt/baselines/baseline_panel.json \
  --output outputs/tropical_cyclone_bt/figures/model_metric_gap_profile.svg
```

The plot defaults to a symlog x-axis (`linthresh = 1`). Pass `--x-scale linear`
to recover the linear layout (and optionally `--x-max VALUE` to clip the
upper bound when an outlier baseline would otherwise dominate). Models are
rendered as filled circles in color; baselines as open diamonds in grayscale.

## Statistical conventions

- **Cluster bootstrap.** IBTrACS storm SIDs are parsed from the packed-patch
  `rel_paths` (`<seq>-<SID>_<YYYYMMDD>T<HHMM>_ml.nc`). The real-vs-real null,
  the corruption sweep, and the model-vs-real bootstrap all resample storms
  with replacement and take every image from each drawn storm. The classical
  iid bootstrap is used only for sides that have no cluster structure (e.g.
  diffusion-generated samples).
- **Class aggregation.** Per-class metric values are macro-averaged with
  equal weight per intensity class. Per-class details are kept in the report.
- **Metric inclusion gate.** A metric is shown on the standardized-gap
  profile when (a) its real-vs-real null has a finite positive width and
  (b) it is responsive: at least one *bad-but-sensible baseline* pushes its
  class-aggregate standardized gap above 1 (the preferred, baseline-panel
  gate). Pass the `baseline_panel.json` produced by
  `evaluate_tc_bt_baselines.py` to the model evaluator and plotter via
  `--baseline-responsiveness` to enable this gate. The legacy
  corruption-validation gate is still available as a fallback when no
  baseline-panel file is provided; corruption-sweep results are otherwise
  retained as audit evidence in the standardization report.
- **Bootstrap calibration audit.** The standardization report includes a
  `bootstrap_calibration` block that runs an independent real-vs-real
  cluster bootstrap and reports, per metric, the empirical fraction of
  replicates that exceed `gap = 1` under H₀. Under correct calibration this
  fraction should sit within a 2σ binomial band around 5%; sustained
  excursions diagnose an under- or over-estimated noise floor.
- **Standardized-gap bootstrap.** Bootstrap intervals on the standardized
  gaps are conditional on a fixed standardization report — i.e. they describe
  finite-sample uncertainty of the model evaluation given the noise floor,
  not joint uncertainty in noise floor and model together. The standardization
  side has its own independent bootstrap reps.

## Notes

The learned evaluator-embedding metrics from the original diffusion repository
are not ported here. They would be an optional module with explicit evaluator
weights and documented training provenance.
