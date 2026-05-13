# climategen-eval

`climategen-eval` is a standalone evaluator for generative climate-field models.
The repository is intentionally independent of diffusion training code: it consumes real
reference data, model-generated sample banks, and emits metric reports,
null-standardization reports, and publication figures.

The initial case study is tropical-cyclone brightness temperature imagery. The current
port includes:

- NumPy metric primitives for histograms, spectra, radial profiles, DAV, coverage/precision
  helpers, and nearest-neighbor summaries.
- A lightweight TC brightness-temperature evaluator for real test data and generated NPY
  sample banks.
- Real-vs-real null utilities that define null-standardized effect sizes for model
  metrics.
- Corruption sweeps for metric responsiveness validation.
- A spider-plot script that reads plain `metrics.json` paths and maps standardized
  gaps onto `[0, 1]` for display.
- A case-study data preparation script that exports only held-out test data and links/copies
  the two generated sample banks used in the current comparison.

## Current Case-Study Sources

From the original `tc-diffusion` workspace:

- Real data split: `data/splits/files_test.txt`
- Real packed data: `data/packed_tc_memmap/manifest.json`
- Metadata: `data/dataset_index_wind.json`
- `pinn_loss` generated bank: `outputs/pinn_loss/sample_banks/test/final_bank`
- `FINAL_BASELINE` generated bank: `outputs/FINAL_BASELINE/sample_banks/test/final_bank`

The `FINAL_BASELINE` bank is inferred from the `final_eval_1000` timestamp: the later
reranked bank was created on 2026-05-01, after `final_eval_1000` was written on
2026-04-30, so it cannot be the source for that eval.

## Quick Start

Prepare a local case-study data directory from the original repo:

```bash
python case_studies/tropical_cyclone_bt/prepare_case_study.py \
  --source-repo ../tc-diffusion \
  --output data/tropical_cyclone_bt \
  --copy-mode symlink
```

Compute the null-standardization report:

```bash
python scripts/compute_tc_bt_standardization.py \
  --config configs/tc_bt_case_study.yaml \
  --generated-n-per-class 1000 \
  --real-reference-reps 200 \
  --output outputs/tropical_cyclone_bt/standardization/test_n1000/metric_standardization.json
```

Evaluate the two sample banks:

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

By default, evaluation writes two-sample within-class bootstrap intervals for
aggregate raw metrics. When a standardization report is provided, it also writes
intervals for standardized gaps and display scores; the spider plot renders
display-score intervals when they are present in the metrics files. Use
`--bootstrap-reps 0` only for quick smoke runs where uncertainty intervals are
not needed.

The standardization report separates two roles:

- `null_standardization`: the primary score ruler. For lower-is-better metrics,
  `good = q50(real-vs-real)` and `scale = q95(real-vs-real) - q50(real-vs-real)`.
  A model's standardized gap is `max(0, model - good) / scale`.
- `metric_validation.corruption_validation`: corruption sweeps used to validate
  that metrics respond to relevant degradations. The report records the
  justifying degradation family, the first severity reaching target power,
  Spearman response-shape correlations, directional consistency, monotonicity,
  and the maximum response in real-vs-real null widths. A metric is included in
  null-standardized scores and the spider plot only when it has a finite
  positive null width and passes this responsiveness check.

Plot null-standardized model scores:

```bash
python scripts/plot_model_metric_summary.py \
  --input FINAL_BASELINE=outputs/tropical_cyclone_bt/eval/FINAL_BASELINE/metrics.json \
  --input pinn_loss=outputs/tropical_cyclone_bt/eval/pinn_loss/metrics.json \
  --standardization-report \
  outputs/tropical_cyclone_bt/standardization/test_n1000/metric_standardization.json \
  --display-null-widths 4 \
  --output outputs/tropical_cyclone_bt/figures/model_metric_summary.svg
```

With the default null-standardized plot mode, a score of zero means the metric is
at least four one-sided real-vs-real null widths worse than the finite-sample ideal.

## Notes

The learned evaluator-embedding metrics from the original diffusion repository are not
ported in this first pass. They should become an optional module with explicit evaluator
weights and documented training provenance.
