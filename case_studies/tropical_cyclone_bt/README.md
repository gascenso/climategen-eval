# Tropical-cyclone brightness-temperature case study

This adapter demonstrates GAUGE v0.2 on storm-centred infrared
brightness-temperature fields. Storm is the independent event. The six native
conditions are TS and Categories 1–5; TD and pooled C4+ conclusions are not
claimed.

## Data and provenance

Every intensity label and continuous conditioning value comes from IBTrACS
`USA_WIND`, a one-minute sustained wind. `WMO_WIND` is rejected because it
mixes agency-specific averaging periods. Real train, validation, and test
partitions are storm disjoint.

The raw archives are retained byte-for-byte. The adapter exposes the same lazy
170–325 K clipped representation used for generator training and records
contact with those boundaries as a **preprocessing saturation audit**, not as a
general physical-admissibility result.

Prepare local artifacts from the sibling diffusion repository with:

```bash
python case_studies/tropical_cyclone_bt/prepare_case_study.py \
  --source-repo ../tc-diffusion --copy-mode hardlink --overwrite
```

The selected generator is called the **TC brightness-temperature diffusion
model** in scientific text. Its internal run identifier, `baseline_ztsnr`,
appears only in reproducibility metadata.

## Versioned analysis protocol

[`analysis_protocol.yaml`](analysis_protocol.yaml) records the analysis and all
method-development amendments. It is not described as an untouched prospective
preregistration: the TC archive informed development before the v0.2 contract
was frozen and rerun.

Primary fidelity conclusions use only TS/C1/C2/C3/C4/C5. The naturally
imbalanced real mixture and class-balanced generated bank are retained in an
as-provided global dataset-composition audit, which is excluded from the
headline fidelity profile.

## Registered evidence

Seven scalar functionals cover core-minimum temperature, cold-cloud area below
220 K, maximum-gradient radius, azimuthal departure, deviation-angle variance,
eye/eyewall contrast, and the first BT percentile. The adapter also defines a
32-bin radial profile and a 24-bin Hann-windowed radial log-PSD discrepancy in
`dex²`.

Training-fitted feature metrics report biased non-negative Gaussian MMD²,
density departure, coverage deficit, and two-sided diversity departure.
Localized memorization is evaluated within matched intensity/feature regions.

## Metric controls

Run the training-only registry with:

```bash
MPLCONFIGDIR=/tmp/mpl-gauge PYTHONPATH=src \
  python case_studies/tropical_cyclone_bt/run_track1_controls.py
```

The output contains complete severity curves and separate sensitivity,
invariance, and resolution fields for temperature, threshold, eye/eyewall,
blur, displacement, radial deformation, phase, PSD, feature-manifold,
preprocessing, and copying controls. It does not collapse them into one
scientific pass/fail claim. The copied-training audit reports 1%, 5%, 10%, 25%,
50%, and 100% mixtures.

## Clean evaluation and secondary audits

```bash
MPLCONFIGDIR=/tmp/mpl-gauge PYTHONPATH=src \
  python -m gauge_climate.cli run \
  case_studies/tropical_cyclone_bt/track1_config.template.yaml

MPLCONFIGDIR=/tmp/mpl-gauge PYTHONPATH=src \
  python case_studies/tropical_cyclone_bt/audit_continuous_wind_response.py

MPLCONFIGDIR=/tmp/mpl-gauge PYTHONPATH=src \
  python case_studies/tropical_cyclone_bt/audit_feature_robustness.py
```

The continuous-wind audit estimates within-class functional slopes without an
equivalence verdict. The feature audit is explicitly post hoc and spans EOF fit
sizes 256/512/1024, ranks 10/20/40, and low-pass EOF scales of 16/32/64 km.

## Frozen example report

After the schema-v3 report exists, freeze its machine-readable and HTML forms
with:

```bash
python scripts/freeze_tc_v02_results.py
```

The checked-in [`final_results`](final_results) directory contains the frozen
evaluation report, complete browser-readable card, and checksum manifest. The
continuous-wind and feature-space audit commands above produce their own JSON,
CSV, and figure outputs from the same declared partitions.
