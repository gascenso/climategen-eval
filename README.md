# climategen-eval

`climategen-eval` is a standalone evaluator for generative climate-field models.
The repository is intentionally independent of diffusion training code: it consumes real
reference data, model-generated sample banks, and emits metric reports, normalization
anchors, and publication figures.

The initial case study is tropical-cyclone brightness temperature imagery. The current
port includes:

- NumPy metric primitives for histograms, spectra, radial profiles, DAV, coverage/precision
  helpers, and nearest-neighbor summaries.
- A lightweight TC brightness-temperature evaluator for real test data and generated NPY
  sample banks.
- Negative-control and anchor utilities matching the current TC diffusion evaluation logic.
- A spider-plot script that reads plain `metrics.json` paths.
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
  --source-repo .. \
  --output data/tropical_cyclone_bt \
  --copy-mode symlink
```

Evaluate the two sample banks:

```bash
python scripts/evaluate_tc_bt_sample_banks.py \
  --config configs/tc_bt_case_study.yaml \
  --model FINAL_BASELINE=data/tropical_cyclone_bt/generated/FINAL_BASELINE/final_bank \
  --model pinn_loss=data/tropical_cyclone_bt/generated/pinn_loss/final_bank \
  --n-per-class 1000 \
  --output-root outputs/tropical_cyclone_bt/eval
```

Compute normalization anchors:

```bash
python scripts/compute_tc_bt_anchors.py \
  --config configs/tc_bt_case_study.yaml \
  --generated-n-per-class 1000 \
  --anchor-mode calibrated \
  --real-reference-reps 200 \
  --calibration-reps 200 \
  --output outputs/tropical_cyclone_bt/anchors/test_n1000_calibrated/metric_normalization.json
```

Plot normalized model scores:

```bash
python scripts/plot_model_metric_summary.py \
  --input FINAL_BASELINE=outputs/tropical_cyclone_bt/eval/FINAL_BASELINE/metrics.json \
  --input pinn_loss=outputs/tropical_cyclone_bt/eval/pinn_loss/metrics.json \
  --normalization-anchors outputs/tropical_cyclone_bt/anchors/test_n1000_calibrated/metric_normalization.json \
  --output outputs/tropical_cyclone_bt/figures/model_metric_summary.svg
```

## Notes

The learned evaluator-embedding metrics from the original diffusion repository are not
ported in this first pass. They should become an optional module with explicit evaluator
weights and documented training provenance.

