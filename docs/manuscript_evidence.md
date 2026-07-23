# GAUGE manuscript evidence map

This is the handoff from framework development to paper writing. Every listed
result is generated from the frozen report; no value needs to be copied from an
exploratory notebook.

## Main-text evidence

| Proposed item | File | Purpose |
|---|---|---|
| Figure 1 | `case_studies/tropical_cyclone_bt/paper_results/main/figure_1_framework_overview.pdf` | Introduce inputs, dependence-matched calibration, evidence families, and interpretation-first output. |
| Figure 2 | `case_studies/tropical_cyclone_bt/paper_results/main/figure_2_statistical_validation.pdf` | Establish false-positive behavior, power, thin-block resolution, and simultaneous-family behavior. |
| Figure 3 | `case_studies/tropical_cyclone_bt/paper_results/main/figure_3_tc_examples.pdf` | Show deterministic, non-cherry-picked typical real/generated fields by native intensity class. |
| Figure 4 | `case_studies/tropical_cyclone_bt/paper_results/main/figure_4_reference_gap_profile.pdf` | Headline class-aware GAUGE result with training-only degraded-data anchors. |
| Figure 5 | `case_studies/tropical_cyclone_bt/paper_results/main/figure_5_tail_physical_distributions.pdf` | Preserve the thin but scientifically important Category-5 functional evidence. |
| Figure 6 | `case_studies/tropical_cyclone_bt/paper_results/main/figure_6_tail_spatial_structure.pdf` | Show Category-5 radial and spectral structure without replacing the scalar detection rule by visual overlap. |
| Table 1 | `case_studies/tropical_cyclone_bt/paper_results/main/table_1_metric_registry.md` | Define every metric, unit, role, orientation, and control. |
| Table 2 | `case_studies/tropical_cyclone_bt/paper_results/main/table_2_native_metric_values.csv` | Provide the exact native-unit values, intervals, reference summaries, gaps, and statuses behind Figure 4. |

The ordering is a recommendation, not a requirement. A venue with a stricter
figure limit can move Figures 3, 5, or 6 to the supplement without losing the
framework argument.

## Appendix and reproducibility evidence

- `appendix/figure_A1_conditional_gap_matrix.pdf`: lossless class-by-metric view.
- `appendix/figure_A2_sensitivity_controls.pdf`: all registered training-only
  corruption/copyist response lanes.
- `appendix/figure_A3_all_class_radial_profiles.pdf` and
  `figure_A4_all_class_radial_psd.pdf`: all native-class curve diagnostics.
- `appendix/figure_A5_generation_audits.pdf`: compact copying and preprocessing
  safeguards.
- `appendix/functional_distributions/`: all seven functionals for all native
  classes.
- `appendix/extended_diagnostics/`: feature, dependence, localized-copying, and
  admissibility audits.
- `appendix/table_A1_complete_evaluation.csv`: every report cell, including
  global composition and supplemental C4+ rows.
- `appendix/table_A2_example_selection.csv`: deterministic montage-selection
  provenance.
- `appendix/table_A3_partition_summary.*`: storms, frames, and classes by split.
- `appendix/table_A4_vmax_support.*`: continuous/discrete conditioning support.
- `appendix/table_A5_c5_values.*`: compact native-unit Category-5 table.
- `paper_results/manifest.json`: checksums for the complete bundle.

## Frozen quantitative statements

- The real archives contain 51,890 training frames from 2,761 storms and 7,011
  test frames from 368 storms; the generated test bank contains 6,000 fields,
  balanced at 1,000 per native class.
- Seven of 56 scalar-functional cells exceed their pointwise real-reference
  boundary; all seven are the unmatched global composition rows. None of the 42
  native-class or seven supplemental C4+ scalar-functional cells does.
- Four global scalar-functional cells exceed the registered simultaneous
  Bonferroni boundaries.
- All 104 registered physical, curve, and feature cells pass the frozen
  split-half boundary-stability audit; the headline class-aware profile contains
  the 78 native-class cells.
- The localized test finds zero copying cells, five cells farther from training,
  and 36 inconclusive cells; the exact-copyist positive control passes.
- The generic validation artifact passes every registered gate. Its thin
  22-block regime deliberately shows that 1,000 frames cannot compensate for
  limited independent-block support.

Use `NO_DEPARTURE_DETECTED` wording exactly. It is not equality, equivalence, or
certification. The global rows diagnose supplied-bank mixture mismatch and must
not be described as conditional-generator failure.

## Scope of remaining work

No additional in-repository analysis or figure generation is required before
drafting the manuscript. Three external archival actions remain: independent
statistical review, permission-cleared deposition with persistent identifiers,
and an independent clean-room reproduction. The survey and one-call
application-specific renderer are explicitly outside this freeze.
`python scripts/audit_release_gates.py --local-only` verifies the frozen local
state.
