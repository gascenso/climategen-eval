# GAUGE implementation and release status

This file distinguishes implemented infrastructure from scientific work that
cannot be completed honestly by code alone.

| Milestone | Implemented now | Remaining release gate |
|---|---|---|
| 1. Scientific specification | explicit estimands, reference-aware conclusions, extensible scope, preregistration templates, card mockup, and frozen TC methods | independent scientific/statistical review; record any amendments before final outcomes |
| 2. Statistical validation | whole-block matched reference, block/frame estimands, matched model/reference resampling intervals, Bonferroni simultaneous reference boundaries, and a passed 300-trial calibration/power artifact covering balanced, unequal, 22-block C5-like, and 20-block designs, crossing CDFs, tails, variance changes, train/test shift, familywise behavior, and fail-closed thin strata | independent statistical review before archival v1.0 |
| 3. Package foundation | `gauge-climate` wheel, `gauge_climate` API, NumPy/memmap, optional xarray/Dask, strict split/sample-id/metadata/materialization checks, validate/run/report CLI, clean-wheel synthetic run in CI | clean-room installation by an independent user |
| 4. Metrics/features/controls | scalar W1, MMD², density, coverage, diversity, entry-wise dependence, admissibility, train-fit EOF features, localized copying, perturbation sweeps, TC radial adapters, frozen TC control runner, passed versioned response/copyist artifact | independent review of the TC control artifact and its reported resolution limits |
| 5. Reporting | versioned JSON schemas, grouped HTML card, native-unit/reference fields, reusable paper-bundle API, all-class/large-K conditional profile policy, positive-control overlay, appendix matrix, lossless CSV, browser landing page, and complete TC manuscript bundle | optional post-paper usability study and one-call paper-bundle CLI integration |
| 6. TC case study | six-class USA_WIND/1-minute contract, corrected train/validation/test archives, selected baseline_ztsnr bank, frozen analysis choices, passed training-only controls and generic statistical gate, clean final report, explicit global-mixture interpretation, label-matched memorization, stable native-class headline profile, explicit thin-C5 drill-down, and regenerated main/appendix evidence | independent audit and permission-cleared external data/sample-bank archive |
| 7. Survey | source-XLSX export, canonical CSV/codebook, generated counts, checksums, blinded deterministic 20% sample, agreement tool, provisional plots | freeze searches; reconcile corpus/four candidate rows; independent recode; full load-bearing verification; baseline-strength coding; flow diagram |
| 8. Release | BSD-3, CI matrix, wheel test, changelog, citation/contribution files, statistical artifact, fail-closed release audit, checksummed final report, and 56-artifact manuscript bundle | archive DOI/PIDs, independent review, and independent clean-room TC reproduction |

The local scientific/software gates are complete:
`python scripts/audit_release_gates.py --local-only` passes. At the current
freeze, the full audit fails only because the large TC artifacts have not yet
received permission-cleared archive locations and persistent identifiers. Do
not flip status flags or replace `PENDING` values merely to make the audit
green.
