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

This is a clean current-engine run from the frozen fields, not a migrated
artifact. Scalar functionals use 2,000 reference draws. Every metric displayed
in the main reference-gap profile uses at least 1,000 draws, and feature
coverage uses 2,000. All 104 registered physical, curve, and feature cells pass
the frozen reference-boundary stability gate; 78 native-class cells appear in
the headline profile. Thirty-one exploratory pairwise-dependence cells and two
discrete preprocessing-rate cells fail that gate, retain their raw audit values,
and correctly receive `NOT_EVALUATED` rather than categorical conclusions.
Configuration hash: `4bf4adb40d792785`.

The localized copying audit contains no copying cell: five supported
label-by-feature-region cells are significantly farther from training and 36
are inconclusive. `FARTHER_FROM_TRAINING` is not interpreted as physical
inadmissibility.

Files:

- `evaluation.json`: canonical `gauge_climate.evaluation.v2` report.
- `evaluation_full.html`: complete evaluation card rendered from that report.
- `manifest.json`: checksums and byte counts for both artifacts.
