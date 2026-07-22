# TC GAUGE paper-facing results

This directory is the output an ordinary GAUGE user should start from. The
statistical machinery remains in the frozen report, while the main directory
contains the concise figures and native-unit table intended for a results
section.

## Main-text artifacts

- `main/figure_1_reference_gap_profile.pdf`: primary intrinsic result. Every
  native class is visible and training-only positive controls are open diamonds.
- `main/figure_2_tail_physical_distributions.pdf`: Category 5 empirical distributions
  for all seven physical functionals.
- `main/figure_3_tail_spatial_structure.pdf`: Category 5 radial-profile and spectral
  drill-downs.
- `main/figure_4_generation_safeguards.pdf`: compact memorization and
  preprocessing-boundary audit.
- `main/table_1_native_metric_values.csv`: exact native-unit estimates,
  intervals, reference summaries, gaps, and statuses for every displayed class.
- `main/table_1_c5_values.md`: compact human-readable Category 5 table for
  direct inspection or manuscript adaptation.
- `main/table_1_c5_values.csv`: the same Category 5 values in machine-readable form.

`index.html` provides a browser-friendly landing page. The appendix retains the
complete class matrix, all report cells, every native-class distribution, and
the extended diagnostics.

## Reporting language

- Gap 0 is the median real-vs-real discrepancy; gap 1 is the pointwise 95%
  real-reference boundary.
- Gap units express detectability at the declared sample size, not physical
  severity, and must not be averaged into a model score.
- A point below 1 supports “no departure detected,” not equality.
- The unmatched global mixture is deliberately absent from the fidelity figure;
  it remains in the complete audit table as a dataset-composition check.

Source report: `evaluation.json`

Source report SHA-256: `162514928163ff8366d4739255233259d40c3e40faed02f7d879626ec0403f45`

Configuration hash: `9d4bdd409257157f`
