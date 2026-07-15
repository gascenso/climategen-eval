# GAUGE implementation and release status

This file distinguishes implemented infrastructure from scientific work that
cannot be completed honestly by code alone.

| Milestone | Implemented now | Remaining release gate |
|---|---|---|
| 1. Scientific specification | revised estimands, two-layer conclusions, bounded-W1 contract, extensible scope, preregistration templates, card mockup | independent scientific/statistical review; freeze TC choices |
| 2. Statistical validation | whole-block matched reference, block/frame estimands, DKW W1 bound, clustered diagnostic, Bonferroni simultaneous reference boundaries, equality/shift/variance/tail/unequal-block/train-test simulations | broaden simulation design in independent review; TC-specific frozen calibration |
| 3. Package foundation | `gauge-climate` wheel, `gauge_climate` API, NumPy/memmap, optional xarray/Dask, strict split/sample-id/metadata/materialization checks, validate/run/report CLI, clean-wheel synthetic run in CI | clean-room installation by an independent user |
| 4. Metrics/features/controls | scalar W1, MMD², density, coverage, diversity, entry-wise dependence, admissibility, train-fit EOF features, localized copying, perturbation sweeps, TC radial adapters | freeze TC feature dimension, tolerances, lane severities; validate each final TC response |
| 5. Reporting | versioned JSON schemas, grouped HTML card, native-unit/tolerance/reference fields, diagnostic profile, curve envelopes/figures, memorization and sensitivity sections | five-person 15-minute interpretation study; polished paper figure design |
| 6. TC Track 1 | six-class scope, TC adapter with physical PSD coordinates, config/prereg templates, deterministic checksummed synthetic fixture, storm-id export/migration, checksum generation, partition-summary and release-audit tooling | prepare real train archive; replace symlinks with archived checksummed banks; select primary model; freeze and run final protocol |
| 7. Survey | source-XLSX export, canonical CSV/codebook, generated counts, checksums, blinded deterministic 20% sample, agreement tool, provisional plots | freeze searches; reconcile corpus/four candidate rows; independent recode; full load-bearing verification; baseline-strength coding; flow diagram |
| 8. Generic Track 2 | eight arms, exact partition/provenance checks, continuous-label matching strata, count/transform parity, per-block results, hierarchical paired contrasts, TOST-style equivalence bounds, determinism audit | broaden toy learning tasks and clean external callback example |
| 9. TC Track 2 | preregistration and whole-storm holdout planner; leakage checks require exact restricted generator provenance | select three storms; perform three generator retrains; freeze wind bins/endpoints/margins; execute 5 seeds/arm/holdout |
| 10. Release | BSD-3, CI matrix, wheel test, changelog, citation/contribution files, statistical artifact, fail-closed release audit | archive DOI/PIDs, independent review, clean-room TC reproduction, generated paper figures/tables |

Current release-audit failure is expected and informative. Run
`python scripts/audit_release_gates.py` to see the external/scientific items that
remain. Do not flip status flags or replace `PENDING` values merely to make the
audit green.
