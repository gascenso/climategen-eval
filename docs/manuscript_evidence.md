# GAUGE v0.2 manuscript evidence map

This file maps manuscript claims to generated artifacts. The schema-v3
evaluation report is authoritative; prose, tables, and figures must not introduce
manually transcribed results.

## Main-text evidence

| Item | Generated file | Evidential role |
|---|---|---|
| GAUGE workflow | `case_studies/tropical_cyclone_bt/paper_results/main/figure_1_framework_overview.pdf` | Introduce the three data roles, event-disjoint empirical reference, registered metrics and controls, and report outputs. |
| Synthetic validation | `case_studies/tropical_cyclone_bt/paper_results/main/figure_2_statistical_validation.pdf` | Report same-population boundary crossings, perturbation response, unequal blocks, and thin-event resolution without formal-test language. |
| TC examples | `case_studies/tropical_cyclone_bt/paper_results/main/figure_3_tc_examples.pdf` | Show deterministic, non-cherry-picked real and generated fields for TS and Categories 1–5. |
| Headline gap profile | `case_studies/tropical_cyclone_bt/paper_results/main/figure_4_reference_gap_profile.pdf` | Present the six native conditions and metric-specific degraded-data signposts. |
| Category 5 physical distributions | `case_studies/tropical_cyclone_bt/paper_results/main/figure_5_tail_physical_distributions.pdf` | Preserve native tail evidence and expose the 22-storm limitation. |
| Category 5 spatial structure | `case_studies/tropical_cyclone_bt/paper_results/main/figure_6_tail_spatial_structure.pdf` | Show radial-profile and log-power behavior beside the registered scalar discrepancies. |
| Metric registry | `case_studies/tropical_cyclone_bt/paper_results/main/table_1_metric_registry.md` | Define units, orientation, estimator, provenance, applicable controls, and interpretation status. |
| Native values | `case_studies/tropical_cyclone_bt/paper_results/main/table_2_native_metric_values.csv` | Supply discrepancies, central comparison ranges, reference quantiles and intervals, reference percentiles and gaps, sample support, and statuses. |
| Prose-facing numbers | `manuscript/gmd/tables/results_macros.tex` | Generate validation rates, TC support counts, mixture shares, and memorization-control resolution directly from schema-v3/v4 reports and control artifacts. |

The structured-review figure remains a separate manuscript artifact generated
from the user-owned literature-review workstream.

## Supplementary and reproducibility evidence

- `appendix/figure_A1_conditional_gap_matrix.pdf`: complete six-class by
  13-metric view, without C4+ pooling.
- `appendix/figure_A2_sensitivity_controls.pdf`: full training-only severity and
  invariance curves, including first reference-range crossing where present.
- `appendix/figure_A3_all_class_radial_profiles.pdf` and
  `figure_A4_all_class_radial_psd.pdf`: all-class curve diagnostics.
- `appendix/figure_A5_generation_audits.pdf`: localized training-similarity and
  preprocessing-saturation summaries.
- `appendix/figure_A_reference_stability.pdf`: paired Monte Carlo intervals,
  delete-group archive sensitivity, and correct-block versus frame-IID
  comparison.
- `appendix/figure_A_continuous_wind_response.pdf` and its JSON/CSV:
  within-class functional slopes against the continuous wind condition.
- `appendix/figure_A_feature_space_robustness.pdf` and its JSON: EOF fit-size,
  rank, and physical low-pass robustness.
- `appendix/functional_distributions/`: every physical functional in all six
  native conditions.
- `appendix/extended_diagnostics/`: feature, dependence, localized-copying, and
  preprocessing-saturation audits.
- `appendix/table_A1_complete_evaluation.csv`: every schema-v3 report result,
  including the separately labelled unmatched global composition audit.
- `appendix/table_A2_example_selection.csv`: deterministic montage-selection
  provenance.
- `appendix/table_A3_partition_summary.*`: frames and storms by split and class.
- `appendix/table_A4_vmax_support.*`: generated and held-out conditioning
  support.
- `appendix/table_A5_c5_values.*`: compact Category 5 native-unit results.
- `paper_results/manifest.json`: byte counts and SHA-256 checksums for the
  complete paper bundle.

## Permitted result language

- `WITHIN_REFERENCE_RANGE`: the median model discrepancy does not exceed the
  empirical real-reference 95th percentile under the declared design. It is not
  equality, equivalence, indistinguishability, adequacy, or a pass.
- `ABOVE_REFERENCE_RANGE`: the median model discrepancy is above that empirical
  percentile. It is not a p-value or an exact significance statement.
- `REFERENCE_UNRESOLVED`: the reference width or numerical stability is
  inadequate; no affirmative interpretation is permitted.
- `NOT_EVALUATED`: the input or metric is unsupported or invalid.

Results must be described as named physical or statistical patterns. Do not
report an aggregate number of “failed cells,” and do not treat the unmatched
global class mixture as conditional model fidelity. Category 5 must remain
separate, with its sample and storm counts shown.

## Deferred external gates

The literature-corpus reconstruction and permanent DOI-bearing code/data
deposition are user-owned workstreams. Until they are complete, the manuscript
must retain explicit integration markers and must not be described as fully
submission-ready. The local release audit deliberately distinguishes these
external gates from reproducible in-repository scientific and software checks.
