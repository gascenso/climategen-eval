# TC GAUGE paper-facing results

This is the complete manuscript figure/table bundle generated from one clean
current-engine report. The main directory contains the framework, statistical
validation, case-study, and headline-result artifacts. The appendix retains the
complete class, curve, sensitivity, copying, dependence, admissibility, data,
and provenance evidence.

## Main-text artifacts

- `main/figure_1_framework_overview.pdf`: GAUGE data, calibration, metric, and
  reporting workflow.
- `main/figure_2_statistical_validation.pdf`: Monte Carlo calibration and power,
  including the 22-block C5-like regime.
- `main/figure_3_tc_examples.pdf`: deterministically selected typical real and
  generated fields for TS and Categories 1--5.
- `main/figure_4_reference_gap_profile.pdf`: primary intrinsic result. Every
  native class is visible and training-only positive controls are open diamonds.
- `main/figure_5_tail_physical_distributions.pdf`: Category 5 empirical distributions
  for all seven physical functionals.
- `main/figure_6_tail_spatial_structure.pdf`: Category 5 radial-profile and spectral
  drill-downs.
- `main/table_1_metric_registry.md`: definitions, units, roles, and registered
  controls for the TC metric suite.
- `main/table_2_native_metric_values.csv`: exact native-unit estimates,
  intervals, reference summaries, gaps, and statuses for every displayed class.

The appendix includes the complete conditional matrix, sensitivity lanes,
all-class radial-profile and PSD figures, copying/preprocessing audits, every
native-class functional distribution, extended feature/dependence diagnostics,
partition and Vmax-support tables, deterministic example selection, and every
report cell.

## Reporting language

- Gap 0 is the median real-vs-real discrepancy; gap 1 is the pointwise 95%
  real-reference boundary.
- Gap units express detectability at the declared sample size, not physical
  severity, and must not be averaged into a model score.
- A point below 1 supports “no departure detected,” not equality.
- A `NOT_EVALUATED` cell retains its raw audit value but supports no categorical
  conclusion.
- The unmatched global mixture is deliberately absent from the fidelity figure;
  it remains in the complete audit table as a dataset-composition check.
- Curve shading is pointwise variability; curve detection uses the registered
  scalar curve discrepancy and is not inferred by visually counting excursions.

Source report: `evaluation.json`

Source report SHA-256: `0deb6105130a97a9a71ecbd0c369f094b3d430a00de4a2877a1eba1cba8e6c01`

Configuration hash: `4bf4adb40d792785`
