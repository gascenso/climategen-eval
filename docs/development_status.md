# GAUGE implementation and release status

This file distinguishes implemented infrastructure from scientific work that
cannot be completed honestly by code alone.

| Milestone | Implemented now | Remaining release gate |
|---|---|---|
| 1. Scientific specification | revised estimands, two-layer conclusions, bounded-W1 contract, extensible scope, preregistration templates, card mockup, frozen TC methods and Track-2 holdouts | independent scientific/statistical review; record any amendments before final outcomes |
| 2. Statistical validation | whole-block matched reference, block/frame estimands, validated cluster-W1 influence interval, finite-sample DKW discrete fallback, Bonferroni simultaneous reference boundaries, 500-trial equality/boundary/shift/variance/tail/thin/unequal-block/train-test validation, frozen 56-cell TC calibration | independent statistical review before archival v1.0; TC control attains direct within-tolerance resolution in 16/56 cells and 80% detection power at the tolerance in 12/56 |
| 3. Package foundation | `gauge-climate` wheel, `gauge_climate` API, NumPy/memmap, optional xarray/Dask, strict split/sample-id/metadata/materialization checks, validate/run/report CLI, clean-wheel synthetic run in CI | clean-room installation by an independent user |
| 4. Metrics/features/controls | scalar W1, MMD², density, coverage, diversity, entry-wise dependence, admissibility, train-fit EOF features, localized copying, perturbation sweeps, TC radial adapters, frozen TC control runner, passed versioned response/copyist artifact | independent review of the TC control artifact and its reported resolution limits |
| 5. Reporting | versioned JSON schemas, grouped HTML card, native-unit/tolerance/reference fields, compact functional card, diagnostic profile, curve/feature/memorization figures, CSV and integrity manifest | five-person 15-minute interpretation study; generalize the compact card beyond the TC renderer |
| 6. TC Track 1 | six-class USA_WIND/1-minute contract, corrected train/validation/test archives, selected baseline_ztsnr bank, frozen tolerances/C4+ analysis, passed controls/calibration, completed and audited final report, explicit global-mixture interpretation, label-matched memorization, paper-ready artifact bundle | independent audit and permission-cleared external sample-bank archive; optionally increase unstable secondary diagnostic budgets without changing primary claims |
| 7. Survey | source-XLSX export, canonical CSV/codebook, generated counts, checksums, blinded deterministic 20% sample, agreement tool, provisional plots | freeze searches; reconcile corpus/four candidate rows; independent recode; full load-bearing verification; baseline-strength coding; flow diagram |
| 8. Generic Track 2 | eight arms, lazy large-archive field views, exact partition/provenance checks, continuous-label matching strata, count/transform parity, per-block results, hierarchical paired contrasts, equivalence bounds, determinism audit | broaden toy learning tasks and external callback examples beyond TC |
| 9. TC Track 2 | concrete optional-TensorFlow residual CNN callback adapted from tc-diffusion, three frozen storm holdouts, exact-Vmax matching, frozen transforms/endpoints/margins, runner/template, and whole-storm leakage gates | perform three restricted generator retrains and execute 5 seeds/arm/holdout |
| 10. Release | BSD-3, CI matrix, wheel test, changelog, citation/contribution files, statistical artifact, fail-closed release audit | archive DOI/PIDs, independent review, clean-room TC reproduction, generated paper figures/tables |

Current release-audit failure is expected and informative. Run
`python scripts/audit_release_gates.py` to see the external/scientific items that
remain. Do not flip status flags or replace `PENDING` values merely to make the
audit green.
