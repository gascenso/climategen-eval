# TC GAUGE paper-facing results

This is the complete manuscript figure/table bundle generated from one clean
schema-v3 report. The main directory contains the framework, validation,
case-study, and headline-result artifacts. The appendix retains the complete
class, curve, control, copying, dependence, preprocessing, robustness, data,
and provenance evidence.

## Main-text artifacts

- `main/figure_1_framework_overview.pdf`: GAUGE data, calibration, metric, and
  reporting workflow.
- `main/figure_2_statistical_validation.pdf`: empirical-reference validation,
  including same-population crossings, perturbation response, unequal blocks,
  and the 22-block C5-like regime.
- `main/figure_3_tc_examples.pdf`: deterministically selected typical real and
  generated fields for TS and Categories 1--5.
- `main/figure_4_reference_gap_profile.pdf`: primary intrinsic result. Every
  native class is visible and training-only positive controls use distinct open
  symbols.
- `main/figure_5_tail_physical_distributions.pdf`: Category 5 empirical distributions
  for all seven physical functionals.
- `main/figure_6_tail_spatial_structure.pdf`: Category 5 radial-profile and spectral
  drill-downs.
- `main/table_1_metric_registry.md`: definitions, units, roles, and registered
  controls for the TC metric suite.
- `main/table_2_native_metric_values.csv`: exact native-unit estimates,
  intervals, reference summaries, gaps, and statuses for every displayed class.

The appendix includes the complete conditional matrix, four legible grouped
control-response figures plus the raw all-control provenance plot, all-class
radial-profile and PSD figures, copying/preprocessing audits, every native-class
functional distribution, extended feature/dependence diagnostics, partition and
Vmax-support tables, deterministic example selection, and every report cell.

## Reporting language

- Gap 0 is the median event-disjoint real-vs-real discrepancy; gap 1 is its
  empirical 95th percentile.
- Gap units express finite-archive reference position, not physical severity,
  and must not be averaged into a model score.
- `WITHIN_REFERENCE_RANGE` and `ABOVE_REFERENCE_RANGE` are descriptive labels,
  not hypothesis-test decisions.
- A `REFERENCE_UNRESOLVED` or `NOT_EVALUATED` cell supports no affirmative
  interpretation.
- The unmatched global mixture is deliberately absent from the fidelity figure;
  it remains in the complete audit table as a dataset-composition check.
- Curve shading shows radius- or wavenumber-specific variability; the registered scalar curve
  discrepancy is reported separately and is not inferred by visually counting
  excursions.

Source report: `evaluation.json`

Source report SHA-256: `2b7f11272ec3788655d28e2149a60478e450dcbfe2a80621a2fe9733064f8e87`

Configuration hash: `c8daf1bdd2c79cf2`
