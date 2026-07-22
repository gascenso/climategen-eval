# Frozen TC GAUGE report

This directory contains the canonical machine-readable Track-1 report and the
complete human-readable evaluation card. Paper-facing figures and tables are in
[`../paper_results/`](../paper_results/).

The report uses the v2 detection-only contract. It contains no application
equivalence verdict: each cell reports the raw discrepancy and matched
resampling interval, real-reference envelope and standardized reference gap,
pointwise detection status, sample/block counts, sensitivity information, and
provenance.

Across the 56 registered scalar-functional cells, 7 exceed the pointwise
real-reference boundary and 49 do not. Four exceed their simultaneous
Bonferroni reference boundaries. The seven pointwise departures are the global
composition-check rows; none of the 42 exact class-conditional cells or seven
predeclared C4+ cells exceeds its pointwise boundary. Non-detection is not a
claim of equality or universal application acceptability.

The v2 artifact was migrated without rerunning the model from the complete
matched model-versus-reference draw distributions stored in the frozen v1
artifact. Its point estimate and interval are now the median and central 95%
interval of those draws. The original configuration hash
`9d4bdd409257157f` is retained as source provenance.

Files:

- `evaluation.json`: canonical `gauge_climate.evaluation.v2` report.
- `evaluation_full.html`: complete evaluation card rendered from that report.
- `manifest.json`: checksums and byte counts for both artifacts.
