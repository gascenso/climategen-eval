# GAUGE GMD manuscript package

This directory contains the complete one-column submission draft prepared with
the official Copernicus Publications LaTeX package, version 7.14 (12 March
2026).

## Overleaf

The easiest route is to upload `../GAUGE_v0.2_GMD_Overleaf.zip` as a new Overleaf
project. The main document is `manuscript.tex`; use pdfLaTeX. The archive
preserves the `tables/` subdirectory and includes the Copernicus class,
configuration, bibliography style, bibliography, the separately compilable
`supplement.tex`, and all figure files.

Some Tectonic-based preview services, including Prism, crash while loading the
FontAwesome support bundled through the official Copernicus class. For those
services, use `../GAUGE_v0.2_GMD_Prism.zip`. It differs only by replacing the two
unused deceased/equal-contribution author-marker icons with empty definitions.
Use the official-class Overleaf archive for submission.

The canonical manuscript deliverables are the generated ZIPs and their LaTeX
sources. Local `manuscript.pdf` previews are intentionally not versioned because
they can lag the current generated tables and figures. The repository's official
`copernicus.cls` is unmodified.

## Rebuild generated manuscript assets

From the repository root:

```bash
python scripts/render_gmd_survey_figure.py  # literature-review figure
python scripts/sync_gmd_generated_artifacts.py
python scripts/build_gmd_tables.py
python scripts/build_gmd_bibliography.py
python scripts/audit_gmd_manuscript.py
python scripts/package_gmd_overleaf.py
```

The TC figures are copied from the checksummed paper bundle under
`case_studies/tropical_cyclone_bt/paper_results/`. The statistical-validation
figure is copied from `validation/statistical_validation_v0.2.pdf`. The survey
source files retain their historical repository name; the literature-review
figure, appendix tables, and bibliography
are generated from their canonical machine-readable sources.

## Supplementary material

The main manuscript contains the scientific findings needed to interpret every
supplementary diagnostic. The extensive all-class distributions, control-severity
curves, stability experiments, and robustness audits are staged in
`supplement.tex` because their combined size is not reasonable for the main article.
They do not introduce findings beyond those summarized in the main text. Code and
data availability, author contributions, competing interests, and acknowledgements
remain in `manuscript.tex`, immediately after the conclusions.

## Submission metadata still requiring author action

These are not scientific-analysis tasks and must not be guessed:

1. Deposit the immutable software release and permission-cleared TC artifacts,
   then replace the provisional availability wording with the resulting DOI(s)
   and add the archive citation(s) to `references.bib`.
2. Confirm all author, affiliation, funding, grant, and contributor metadata.
3. Select a key figure; `fig01.pdf` is the natural graphical-abstract choice.
4. Paste `short_summary.txt` into the GMD submission form.

The manuscript correctly describes the 69-paper evidence base as a structured review
of a curated corpus rather than a probability sample, so none of its counts is
presented as a field-wide prevalence estimate. The canonical review protocol
still marks the search freeze and independent recoding as incomplete. Those
gates must be completed and the generated counts re-audited before submission;
the present manuscript wording must not be used to imply that this work is
already complete.
