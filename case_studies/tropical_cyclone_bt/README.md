# Tropical-cyclone brightness-temperature case study

This adapter demonstrates GAUGE on storm-centered infrared brightness-temperature
fields. The dependence block is the storm. The six represented intensity classes
are TS and Categories 1–5; TD is not claimed.

## Evidence state

- The v0.1 TC adapter implements seven scalar functionals, physical radial
  profiles, and a fully specified Hann-windowed physical-wavenumber PSD.
- Existing sample banks and exploratory outputs predate the frozen GAUGE
  reference/inference engine. They are development inputs, not paper results.
- Raw real archives are retained byte-for-byte, while the TC adapter exposes a
  lazy 170--325 K clipped analysis view matching `tc-diffusion` training
  preprocessing. These quantiles are not mislabeled as physical validity limits;
  boundary saturation is reported separately.
- Every intensity label and continuous Vmax target must come from IBTrACS
  `USA_WIND`, whose averaging period is one minute. `WMO_WIND` is prohibited
  because it mixes agency-specific averaging periods. Preparation fails closed
  if this provenance is absent or inconsistent with the Saffir-Simpson class.
- [`preregistration.yaml`](preregistration.yaml) freezes tolerances,
  perturbation severities and sampling, three Track-2 storms, transforms,
  endpoints, and task margins without examining final GAUGE test outcomes.
  The training-only sensitivity/copyist controls and TC statistical calibration
  pass their registered gates. A perfect-generator control is directly placed
  within tolerance in 16/56 cells; 12/56 attain at least 80% shift-detection
  power at the exact tolerance. These are design properties, not model scores.
- Local generated banks are copied or hardlinked from the selected
  `baseline_ztsnr` run. The release audit fails until archived artifacts with
  URLs and SHA-256 checksums replace this local-only provenance.

## Track 1

The final run uses the corrected storm-disjoint train/validation/test partitions
and the `baseline_ztsnr` test sample bank. The training split is required for
train-fit features and the three-sample memorization test; validation is reserved
for model selection; test is used for final Track-1 claims. Category 5 is
predeclared as thin; C4+ is the only permitted aggregate, with its
major-hurricane-threshold rationale frozen before evaluation.

Prepare the local data, attempting hardlinks first and falling back to copies
when the source and destination are on different filesystems:

```bash
python case_studies/tropical_cyclone_bt/prepare_case_study.py \
  --source-repo ../tc-diffusion --copy-mode hardlink --overwrite
```

The script writes version-2 real manifests that record `USA_WIND`, the one-minute
averaging period, source-manifest hashes, per-file hashes, storm IDs, and class
counts. It refuses to silently upgrade an older WMO-labeled archive.

Use `gauge_climate.adapters.tc_bt` for the registered functionals and curves.
Fit standardization, EOF/PCA, bandwidths, and every sensitivity/control lane on
training or explicitly reserved real blocks only.

After all three real splits are prepared, generate and audit Table 1 inputs with
`python case_studies/tropical_cyclone_bt/summarize_partitions.py`.

Audit the discrete Vmax conditioning support in both selected sample banks with
`python case_studies/tropical_cyclone_bt/audit_vmax_support.py`. The generated
conditioning distribution is compared with the corresponding corrected real
split, so common operationally reported wind values are not mistaken for model
collapse.

Execute the frozen training-only control suite before unlocking the final test
run:

```bash
python case_studies/tropical_cyclone_bt/run_track1_controls.py
```

The runner uses 80 deterministically selected training storms, a block-balanced
control size of 1,000, 500 real-reference resamples, and the train-only feature
fit. It writes the versioned JSON artifact and severity-response figure under
`control_results/`. A failed response contract withholds interpretation; failure
to clear the reference boundary is retained separately as resolution information
and is not silently retuned.

The versioned [control artifact](control_results/sensitivity_lanes.json) passes
all five response contracts without using test fields. Its resolution fields
remain part of the result: small center jitter and gradient-radius response to
the frozen blur sweep do not clear the real-reference boundary at N=1,000.

Run the frozen training-only statistical calibration with:

```bash
PYTHONPATH=src python case_studies/tropical_cyclone_bt/run_track1_calibration.py
```

The versioned [calibration artifact](calibration_results/track1_statistics.json)
passes all registered reference-mechanics gates without reading test fields.
Twelve of 56 stratum-functional cells attain at least 80% shift-detection power
at their exact practical tolerance. This is a training-only design-sensitivity
diagnosis, not a model score, a failed null calibration, or permission to relax
tolerances. Separately, the validated whole-block W1 interval places the
training-only pseudo-generator within tolerance in 16 of 56 cells; the discrete
maximum-gradient-radius functional retains the finite-sample DKW fallback. All
registered gates now pass and the frozen Track-1 run is authorized.

## Frozen Track-1 result

The completed machine report, CSV table, integrity manifest, and paper-ready
figures are archived in [`final_results/`](final_results/). The first output
audit exposed an estimand distinction already present in the frozen manifests:
the generated bank is balanced at 1,000 fields per class, whereas the real test
archive has its natural intensity mixture. Global rows are therefore retained
as supplied-bank composition checks; native-class and C4+ rows answer
conditional-generator fidelity.

In the first completed audit, all 49 native-class/C4+ scalar-functional cells
had no pointwise or simultaneous departure detected. All seven pointwise
functional detections occurred in the global composition row. Across the full
original 56-cell family, practical statuses were 13 within tolerance, 7 above
tolerance, 13 inconclusive, and 23 underpowered. This stricter layer is not
replaced by non-detection. Memorization is tested within joint intensity-class
and training-fitted feature-region cells; “farther from training” is never
treated as proof of physical invalidity.

Forty-four secondary cells in the first completed audit, mostly pairwise
Spearman entries, carry a split-half Monte Carlo-stability warning at the frozen
200-draw diagnostic budget. They remain in the machine report with daggers and
are not headline claims; no primary functional cell has this warning.

## Track 2

Vmax regression is primary and category classification secondary. Three
storm-level holdouts require three independent restricted generator fits. A
storm is removed in full even when its frames span intensity categories. The
generic ladder keeps continuous task labels separate from frozen wind-matching
strata, permitting count-matched copy/generated/restored arms.

The headline contrast is generated+geometry versus copy+geometry (`A5−A3`).
Generated versus copy without geometry (`A4−A2`) isolates source effects;
restored-real+geometry versus copy+geometry (`A7−A3`) is empirical headroom,
not a theoretical upper bound. Each holdout is shown separately alongside the
hierarchical paired summary.

The included `baseline_ztsnr` training bank is useful only for development. It
was produced by a generator trained on the complete training split and therefore
cannot enter the final holdout contrasts. Final Track 2 still requires three
restricted generator fits and three corresponding generated banks.

Select candidate holdouts directly from the corrected training manifest, before
examining downstream results:

```bash
python case_studies/tropical_cyclone_bt/plan_track2_holdouts.py \
  --real-manifest data/tropical_cyclone_bt/real_train/manifest.json \
  --eligible-label 4 --eligible-label 5 --require-label 5 \
  --minimum-frames 20 --minimum-eligible-frames 10
```

The frozen seed selected storms `2006259N19155`, `2016273N13300`, and
`2018269N11220` from 97 eligible storms. Added copy, generated, and restored-real
rows are matched at the exact reported Vmax value—not merely by intensity
category—because Vmax regression is primary. Each selected storm's Vmax support
also occurs in its restricted training partition.

`downstream_cnn.py` provides the concrete TensorFlow task callback, adapted from
`tc-diffusion/tc_diffusion/evaluator`. It trains a residual CNN with Vmax and
native-category heads, uses the fixed validation split for checkpoint selection,
and reports metrics separately for every withheld storm. Its architecture and
training defaults follow `evaluator_native.yaml`; GAUGE deliberately selects the
checkpoint by storm-balanced validation Vmax MAE because Vmax regression is the
predeclared primary endpoint. TensorFlow remains an optional dependency
(`gauge-climate[tc-downstream]`). The ladder keeps the large real archive
memory-mapped and materializes only minibatches and the matched added samples.

`run_track2.py` connects that callback to all eight GAUGE arms:

```bash
python -m case_studies.tropical_cyclone_bt.run_track2 \
  case_studies/tropical_cyclone_bt/track2_config.yaml
```

The template intentionally fails until each holdout has a leakage-free restricted
generator bank and an exact training-storm provenance manifest. Copy
`generator_provenance.template.json` for each fit and populate its complete
restricted storm list plus split, configuration, checkpoint, and bank hashes.

## Reproduction gates

```bash
python scripts/audit_release_gates.py
python case_studies/tropical_cyclone_bt/plan_track2_holdouts.py --help
```

The first command is expected to fail until the remaining scientific-review and
artifact gates are genuinely complete. The literature survey is currently a
separate workstream; add `--include-survey` only for the full manuscript audit.
Do not bypass failed software/TC gates for a release.
