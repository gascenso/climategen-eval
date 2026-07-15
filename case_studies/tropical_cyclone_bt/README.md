# Tropical-cyclone brightness-temperature case study

This adapter demonstrates GAUGE on storm-centered infrared brightness-temperature
fields. The dependence block is the storm. The six represented intensity classes
are TS and Categories 1–5; TD is not claimed.

## Evidence state

- The v0.1 TC adapter implements seven scalar functionals, physical radial
  profiles, and a fully specified Hann-windowed physical-wavenumber PSD.
- Existing sample banks and exploratory outputs predate the frozen GAUGE
  reference/inference engine. They are development inputs, not paper results.
- [`preregistration.yaml`](preregistration.yaml) deliberately keeps tolerances,
  the primary generator, perturbation severities, three Track-2 storms, and task
  margins unresolved. A final evaluation is prohibited until those choices are
  justified and frozen without examining final test outcomes.
- Current generated-bank directories are absolute local symlinks. The release
  audit fails until archived artifacts with URLs and SHA-256 checksums replace
  them. [`artifacts.yaml`](artifacts.yaml) records this gate.

## Track 1

The final run must use storm-disjoint train/test partitions and publish a
storm/frame/class table. It reports the baseline and physics-informed variants,
with one primary model frozen before the test run. Category 5 is predeclared as
thin; C4+ is the only permitted aggregate, and only if its scientific rationale
is completed before evaluation.

Use `gauge_climate.adapters.tc_bt` for the registered functionals and curves.
Fit standardization, EOF/PCA, bandwidths, and every sensitivity/control lane on
training or explicitly reserved real blocks only.

After both real splits are prepared, generate and audit Table 1 inputs with
`python case_studies/tropical_cyclone_bt/summarize_partitions.py`.

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

## Reproduction gates

```bash
python scripts/audit_release_gates.py
python case_studies/tropical_cyclone_bt/plan_track2_holdouts.py --help
```

The first command is expected to fail until the preregistration, survey, and
artifact gates are genuinely complete. Do not bypass it for a release.
