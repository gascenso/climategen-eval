# GAUGE implementation and release status

This file distinguishes implemented infrastructure from scientific work that
cannot be completed honestly by code alone.

| Milestone | Implemented now | Remaining release gate |
|---|---|---|
| 1. Scientific specification | revised estimands, two-layer conclusions, bounded-W1 contract, extensible scope, preregistration templates, card mockup, frozen TC methods and Track-2 holdouts | independent scientific/statistical review; record any amendments before final outcomes |
| 2. Statistical validation | whole-block matched reference, block/frame estimands, DKW W1 bound, clustered diagnostic, Bonferroni simultaneous reference boundaries, equality/shift/variance/tail/unequal-block/train-test simulations | broaden simulation design in independent review; TC-specific frozen calibration |
| 3. Package foundation | `gauge-climate` wheel, `gauge_climate` API, NumPy/memmap, optional xarray/Dask, strict split/sample-id/metadata/materialization checks, validate/run/report CLI, clean-wheel synthetic run in CI | clean-room installation by an independent user |
| 4. Metrics/features/controls | scalar W1, MMD², density, coverage, diversity, entry-wise dependence, admissibility, train-fit EOF features, localized copying, perturbation sweeps, TC radial adapters, frozen TC control runner, passed versioned response/copyist artifact | independent review of the TC control artifact and its reported resolution limits |
| 5. Reporting | versioned JSON schemas, grouped HTML card, native-unit/tolerance/reference fields, diagnostic profile, curve envelopes/figures, memorization and sensitivity sections | five-person 15-minute interpretation study; polished paper figure design |
| 6. TC Track 1 | six-class USA_WIND/1-minute contract, corrected local train/validation/test archives, selected baseline_ztsnr bank, frozen tolerances/C4+ analysis, TC adapter with physical PSD coordinates, deterministic fixture, checksums, partition/Vmax-support audits, passed training-only controls | pass TC-specific statistical calibration; archive permission-cleared artifacts; then run the final protocol once |
| 7. Survey | source-XLSX export, canonical CSV/codebook, generated counts, checksums, blinded deterministic 20% sample, agreement tool, provisional plots | freeze searches; reconcile corpus/four candidate rows; independent recode; full load-bearing verification; baseline-strength coding; flow diagram |
| 8. Generic Track 2 | eight arms, lazy large-archive field views, exact partition/provenance checks, continuous-label matching strata, count/transform parity, per-block results, hierarchical paired contrasts, equivalence bounds, determinism audit | broaden toy learning tasks and external callback examples beyond TC |
| 9. TC Track 2 | concrete optional-TensorFlow residual CNN callback adapted from tc-diffusion, three frozen storm holdouts, exact-Vmax matching, frozen transforms/endpoints/margins, runner/template, and whole-storm leakage gates | perform three restricted generator retrains and execute 5 seeds/arm/holdout |
| 10. Release | BSD-3, CI matrix, wheel test, changelog, citation/contribution files, statistical artifact, fail-closed release audit | archive DOI/PIDs, independent review, clean-room TC reproduction, generated paper figures/tables |

Current release-audit failure is expected and informative. Run
`python scripts/audit_release_gates.py` to see the external/scientific items that
remain. Do not flip status flags or replace `PENDING` values merely to make the
audit green.
