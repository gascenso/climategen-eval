# GAUGE survey pipeline

The XLSX in `docs/methods/` is the supplied coding source. Regenerate reviewable
artifacts with:

```bash
python survey/export_survey.py
```

This exports the 69 coded rows, codebook, unresolved candidate list, a blinded
deterministic 20% recoding sample, recomputed strict/lenient counts, and SHA-256
manifest. It never reads headline values from the spreadsheet's Summary sheet.

After an independent coder completes the blinded CSV:

```bash
python survey/agreement.py --recoded path/to/completed_double_coding.csv
```

The current artifacts are **not paper-final**. `protocol.yaml` records the
unfrozen queries, corpus reconciliation, downstream-baseline field, and human
verification gates. Until those fields are complete, `summary.json` keeps
`headline_use_permitted: false`.

Provisional, visibly watermarked figures can be regenerated with
`python survey/plot_survey.py`. A PRISMA-style flow is intentionally not
fabricated: it remains unavailable until a frozen record-level screening log
exists.
