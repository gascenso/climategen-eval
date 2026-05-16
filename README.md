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
5. **A bad-but-sensible baseline panel.** Four parameter-free reference
   procedures with textbook interpretations are evaluated alongside the model:
   - **CLIM** — per-class pixel-wise climatological mean (the WMO no-skill
     forecast). Saturates variance- and structure-sensitive metrics.
   - **GAUSS-PIX** — per-class mean plus iid Gaussian noise at the per-pixel
     std. The MLE under independent-Gaussian pixels. Correct first two pixel
     moments, wrong spatial structure.
   - **CROSS-CLASS** — real images drawn from the other intensity classes.
     The class-conditional-independence null; tests label fidelity.
   - **PHASE-SHUFFLE** — Theiler-style FFT phase randomization under
     Hermitian symmetry. Preserves per-image radial PSD and pixel histogram,
     destroys coherent spatial structure.
   These complementary baselines anchor the "decidedly not good enough" side
   of the scale: a model strictly inside the baseline cluster on every metric
   is doing real work; one mixing with the baselines on some metrics has work
   left.

The final figure overlays models and baselines on the same standardized-gap
axis. The x-axis defaults to a symmetric-log scale with `linthresh = 1` so the
noise-floor band stays in linear resolution while baselines that are dozens or
hundreds of widths out do not squash the model region.

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

Evaluate the diffusion sample banks against that report:

```bash
python scripts/evaluate_tc_bt_sample_banks.py \
  --config configs/tc_bt_case_study.yaml \
  --model FINAL_BASELINE=data/tropical_cyclone_bt/generated/FINAL_BASELINE/final_bank \
  --model pinn_loss=data/tropical_cyclone_bt/generated/pinn_loss/final_bank \
  --n-per-class 1000 \
  --standardization-report \
    outputs/tropical_cyclone_bt/standardization/test_n1000/metric_standardization.json \
  --output-root outputs/tropical_cyclone_bt/eval
```

Evaluate the bad-but-sensible baseline panel on the same data:

```bash
python scripts/evaluate_tc_bt_baselines.py \
  --config configs/tc_bt_case_study.yaml \
  --n-per-class 1000 \
  --standardization-report \
    outputs/tropical_cyclone_bt/standardization/test_n1000/metric_standardization.json \
  --output-root outputs/tropical_cyclone_bt/baselines
```

Render the final standardized-gap figure with models and baselines on the same
axis:

```bash
python scripts/plot_model_metric_summary.py \
  --input FINAL_BASELINE=outputs/tropical_cyclone_bt/eval/FINAL_BASELINE/metrics.json \
  --input pinn_loss=outputs/tropical_cyclone_bt/eval/pinn_loss/metrics.json \
  --baseline CLIM=outputs/tropical_cyclone_bt/baselines/clim/metrics.json \
  --baseline GAUSS-PIX=outputs/tropical_cyclone_bt/baselines/gauss_pix/metrics.json \
  --baseline CROSS-CLASS=outputs/tropical_cyclone_bt/baselines/cross_class/metrics.json \
  --baseline PHASE-SHUFFLE=outputs/tropical_cyclone_bt/baselines/phase_shuffle/metrics.json \
  --standardization-report \
    outputs/tropical_cyclone_bt/standardization/test_n1000/metric_standardization.json \
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
- **Metric inclusion gate.** A metric is shown on the standardized-gap profile
  only when (a) its real-vs-real null has a finite positive width and (b) at
  least one corruption family drives it past the null boundary at the
  configured power threshold (default 0.80). Other metrics are kept in the
  raw report but not on the gap axis.
- **Standardized-gap bootstrap.** Bootstrap intervals on the standardized
  gaps are conditional on a fixed standardization report — i.e. they describe
  finite-sample uncertainty of the model evaluation given the noise floor,
  not joint uncertainty in noise floor and model together. The standardization
  side has its own independent bootstrap reps.

## Notes

The learned evaluator-embedding metrics from the original diffusion repository
are not ported here. They would be an optional module with explicit evaluator
weights and documented training provenance.
