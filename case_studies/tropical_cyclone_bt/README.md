# Tropical-cyclone brightness-temperature case study

This adapter demonstrates GAUGE on storm-centered infrared brightness-temperature
fields. The dependence block is the storm. The six represented intensity classes
are TS and Categories 1–5; TD is not claimed.

## Evidence state

- The v0.1 TC adapter implements seven scalar functionals, physical radial
  profiles, and a fully specified Hann-windowed physical-wavenumber PSD.
- The selected sample bank has been evaluated by the frozen current engine.
  Earlier exploratory outputs are not used as paper evidence.
- Raw real archives are retained byte-for-byte, while the TC adapter exposes a
  lazy 170--325 K clipped analysis view matching `tc-diffusion` training
  preprocessing. These quantiles are not mislabeled as physical validity limits;
  boundary saturation is reported separately.
- Every intensity label and continuous Vmax target must come from IBTrACS
  `USA_WIND`, whose averaging period is one minute. `WMO_WIND` is prohibited
  because it mixes agency-specific averaging periods. Preparation fails closed
  if this provenance is absent or inconsistent with the Saffir-Simpson class.
- [`preregistration.yaml`](preregistration.yaml) freezes the functionals,
  perturbation severities, sampling policy, conditions, and analysis exclusions
  without examining final GAUGE test outcomes.
  The training-only sensitivity/copyist controls and generic real-reference
  validation pass their registered gates.
- Local generated banks are copied or hardlinked from the selected
  `baseline_ztsnr` run. The release audit fails until archived artifacts with
  URLs and SHA-256 checksums replace this local-only provenance.

## Intrinsic GAUGE evaluation

The final run uses the corrected storm-disjoint train/validation/test partitions
and the `baseline_ztsnr` test sample bank. The training split is required for
train-fit features and the three-sample memorization test; validation is reserved
for model selection; test is used for final GAUGE claims. Category 5 is
predeclared as thin. The completed audit retains the originally registered C4+
supplemental row, but it is an extreme-tail descriptive aggregate—not the
standard major-hurricane grouping—and never replaces separate C4/C5 results.

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

Audit the discrete Vmax conditioning support in the selected test sample bank with
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

The generic Monte Carlo validation under `validation/` checks reference-plan
invariants, pointwise and simultaneous error behavior, unequal block sizes, and
controlled-shift power. Together with the TC controls, these gates authorize the
frozen evaluation.

## Frozen evaluation result

The completed machine report, CSV table, integrity manifest, and paper-ready
figures are archived in [`final_results/`](final_results/). The first output
audit exposed an estimand distinction already present in the frozen manifests:
the generated bank is balanced at 1,000 fields per class, whereas the real test
archive has its natural intensity mixture. Global rows are therefore retained
as supplied-bank composition checks; native-class rows answer
conditional-generator fidelity. The registered C4+ row remains supplemental.

In the clean final audit, all 49 native-class plus supplemental-C4+ scalar-functional
cells had no pointwise departure detected. All seven pointwise functional
detections occurred in the global composition row, and four of those global
cells also exceeded the registered simultaneous boundaries. Memorization is
tested within joint
intensity-class and training-fitted feature-region cells; “farther from
training” is never treated as proof of physical invalidity.

All 104 registered physical, curve, and feature cells—including the global and
supplemental C4+ audits—pass the frozen reference-boundary stability check and
receive categorical detection results. The main class-aware profile shows the
78 native-class cells.
Thirty-one exploratory pairwise-Spearman cells and two discrete
preprocessing-rate cells fail that check; they retain raw audit values but
correctly receive `NOT_EVALUATED`. No main-profile conclusion is withheld.

## Paper-facing bundle

[`paper_results/`](paper_results/) is the recommended entry point for reading or
reporting the case study. It separates a small main-text package from a complete
appendix:

- a framework schematic, generic statistical-validation figure, and
  deterministically selected real/generated example montage;
- one reference-calibrated profile containing all TS/C1--C5 results and
  training-only positive controls, without a pooled tail headline;
- Category 5 physical-distribution and spatial-structure drill-downs, explicitly
  marked as thin-stratum evidence;
- all-class curve figures, a compact memorization/admissibility safeguard panel,
  and extended feature/dependence audits;
- a complete class-by-metric matrix, partition/Vmax tables, deterministic
  montage-selection audit, and lossless native-unit CSV tables;
- per-functional appendix figures containing all six native-class distributions.

The unmatched global composition result remains in the machine audit and full
table but is deliberately absent from the conditional-fidelity figure. Rebuild
the bundle from the frozen report with:

```bash
PYTHONPATH=src python scripts/render_tc_track1_paper.py \
  case_studies/tropical_cyclone_bt/final_results/evaluation.json \
  --config case_studies/tropical_cyclone_bt/track1_config.template.yaml \
  --controls case_studies/tropical_cyclone_bt/control_results/sensitivity_lanes.json \
  --output-dir case_studies/tropical_cyclone_bt/paper_results
```

## Reproduction gates

```bash
python scripts/audit_release_gates.py --local-only
python scripts/audit_release_gates.py
```

The first command verifies the complete in-repository evidence and must pass.
The second is expected to fail only until the remaining external
scientific-review and artifact-deposition gates are genuinely complete. The
literature survey is currently a
separate workstream; add `--include-survey` only for the full manuscript audit.
Do not bypass failed software/TC gates for a release.
