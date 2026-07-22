# GAUGE implementation and release status

This file distinguishes implemented infrastructure from scientific work that
cannot be completed honestly by code alone.

| Milestone | Implemented now | Remaining release gate |
|---|---|---|
| 1. Scientific specification | explicit estimands, reference-aware conclusions, extensible scope, preregistration templates, card mockup, and frozen TC methods | independent scientific/statistical review; record any amendments before final outcomes |
| 2. Statistical validation | whole-block matched reference, block/frame estimands, matched model/reference resampling intervals, Bonferroni simultaneous reference boundaries, and a passed 500-trial equality/shift/variance/tail/unequal-block validation artifact | independent statistical review before archival v1.0; expand validation to additional thin-stratum and distribution-shift regimes |
| 3. Package foundation | `gauge-climate` wheel, `gauge_climate` API, NumPy/memmap, optional xarray/Dask, strict split/sample-id/metadata/materialization checks, validate/run/report CLI, clean-wheel synthetic run in CI | clean-room installation by an independent user |
| 4. Metrics/features/controls | scalar W1, MMD², density, coverage, diversity, entry-wise dependence, admissibility, train-fit EOF features, localized copying, perturbation sweeps, TC radial adapters, frozen TC control runner, passed versioned response/copyist artifact | independent review of the TC control artifact and its reported resolution limits |
| 5. Reporting | versioned JSON schemas, grouped HTML card, native-unit/reference fields, reusable paper-bundle API, all-class/large-K conditional profile policy, positive-control overlay, appendix matrix, lossless CSV and browser landing page | five-person 15-minute interpretation study; add the paper-bundle declaration to the CLI configuration schema |
| 6. TC case study | six-class USA_WIND/1-minute contract, corrected train/validation/test archives, selected baseline_ztsnr bank, frozen analysis choices, passed training-only controls and generic statistical gate, completed and audited final report, explicit global-mixture interpretation, label-matched memorization, native-class headline profile, and explicit thin-C5 drill-down; the originally registered C4+ aggregate is retained only as supplemental audit evidence | independent audit and permission-cleared external sample-bank archive; optionally increase unstable secondary diagnostic budgets without changing primary claims |
| 7. Survey | source-XLSX export, canonical CSV/codebook, generated counts, checksums, blinded deterministic 20% sample, agreement tool, provisional plots | freeze searches; reconcile corpus/four candidate rows; independent recode; full load-bearing verification; baseline-strength coding; flow diagram |
| 8. Release | BSD-3, CI matrix, wheel test, changelog, citation/contribution files, statistical artifact, fail-closed release audit | archive DOI/PIDs, independent review, clean-room TC reproduction, generated paper figures/tables |

Current release-audit failure is expected and informative. Run
`python scripts/audit_release_gates.py` to see the external/scientific items that
remain. Do not flip status flags or replace `PENDING` values merely to make the
audit green.
