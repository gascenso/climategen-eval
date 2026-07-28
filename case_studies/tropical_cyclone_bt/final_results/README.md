# Frozen TC GAUGE v0.2 report

This directory contains the canonical schema-v3 TC evaluation and its complete
human-readable card. Paper-facing figures, tables, robustness analyses, and
checksums are in [`../paper_results/`](../paper_results/).

The report follows the empirical reference-calibration contract. Every result
records:

- the native model discrepancy and central comparison range over admissible
  held-out real subsets;
- the empirical real-reference median and 95th percentile, with conditional
  Monte Carlo intervals;
- the reference percentile and gap, with a paired-bootstrap interval;
- the sample count, independent-storm count, block-balanced estimand, metric
  and preprocessing provenance, warnings, reference status, and control status.

The six native conditions are TS and Categories 1–5. There is no C4+ aggregate.
The unmatched as-provided global comparison is retained only as a
dataset-composition audit and is excluded from conditional-fidelity statements.
Category 5 remains visible with its 22-storm limitation.

`WITHIN_REFERENCE_RANGE` and `ABOVE_REFERENCE_RANGE` describe position relative
to the study-specific empirical reference. They are not equality, equivalence,
significance, or aggregate pass/fail statements. `REFERENCE_UNRESOLVED` fails
closed when the reference width or numerical stability is inadequate.

The localized memorization analysis uses all 51,890 diffusion-training fields as
its nearest-neighbour reference, while the EOF representation and MMD bandwidth
are fitted only on the declared 512-field training subset. Its regional central
ranges are descriptive and are accompanied by exact-replay and copied-fraction
controls.

Files:

- `evaluation.json`: canonical `gauge_climate.evaluation.v3` report generated
  by `gauge-climate` 0.2.0.
- `evaluation_full.html`: complete evaluation card rendered from that report.
- `manifest.json`: schema-v2 checksums, byte counts, package version, and
  configuration hash for both artifacts.
