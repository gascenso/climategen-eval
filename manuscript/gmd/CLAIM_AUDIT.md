# Manuscript claim audit

The manuscript separates three kinds of evidence.

| Claim class | Authoritative source | Manuscript use |
|---|---|---|
| Protocol definitions and permitted wording | `docs/methods/gauge_blueprint.md`, package source, and statistical tests | Methods and interpretation |
| Statistical operating characteristics | `validation/statistical_validation_v0.2.json` | Statistical validation |
| TC data, model, and results | `case_studies/tropical_cyclone_bt/final_results/evaluation.json` and the checksummed `paper_results/` bundle | Case-study methods, results, figures, and native table |
| Literature-review counts | `survey/exports/summary.json` and `survey/exports/coded_papers.csv` | Curated-corpus evidence; not a field-wide prevalence estimate |
| Prior-work and background claims | `references.bib`, built from DOI metadata and manually verified foundational sources | Introduction and method attribution |

Run `python scripts/audit_gmd_manuscript.py` from the repository root. The audit
checks all citation keys and cross-references, GMD source restrictions, figure
presence, the six-condition metric registry and descriptive reference statuses,
the global-mixture caveat, memorization resolution, validation rates, partition
counts, and every literature-review count quoted in the text. It does not turn
the independently interpreted metric-by-condition results into an aggregate
pass/fail count.

The audit cannot replace expert judgment about whether the TC physical
quantities are scientifically sufficient, complete the outstanding
literature-review protocol gates, create archive DOIs, or verify funding
metadata. Those remain explicit author/release actions.
