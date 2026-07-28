# GAUGE submission stress-test matrix

This matrix maps the two synthetic referee reports in
`manuscript/Peer review of GAUGE.docx` to concrete v0.2 changes. It is an
internal readiness record, not a response letter to an actual journal review.

| Referee concern | Decision | Scientific/manuscript response | Code or artifact |
|---|---|---|---|
| The q95 rule is not an exact level-5% test | Accept | Recast the method as empirical finite-archive reference calibration. Remove significance, equality, indistinguishability, false-positive, and power claims. | schema v3; `docs/statistical_contract.md`; `docs/methods/gauge_blueprint.md` |
| q50, q95, and gap uncertainty absent | Accept | Add paired-bootstrap conditional Monte Carlo intervals for both quantiles and the gap. Distinguish these from archive, retraining, and physical uncertainty. | `reference_summary_uncertainty`; every schema-v3 cell |
| “Noise floor”, oracle, and best-performance language | Accept | Replace with “empirical real-reference distribution” and explicitly reject all three interpretations. | contract, reports, figures, manuscript terminology audit |
| Real-reference and model comparisons have asymmetric dependence | Accept as an estimand clarification, not a reason to use frame-IID real sampling | State that the object audited is an independent generated field bank against the event-balanced real population. Validate the deployed i.i.d. generator versus event-correlated real design; separately show how wrong frame-IID real identifiers narrow/change the reference. | scalar validation `_trial`; block-misspecification validation; revised methods |
| Validation limited to scalar Gaussian W1 | Accept | Add heavy-tailed/multimodal scalar W1, curves, log PSD, MMD², density, coverage, diversity, unequal/thin blocks, bank sizes, shifts, and wrong-block examples. | `scripts/validate_gauge_statistics.py`; validation JSON/PDF |
| “False positive” and “power” labels overclaim | Accept | Rename to same-population boundary-crossing rate and perturbation crossing probability. | validation code, artifact, captions, manuscript |
| Supporting metrics treated as unadjusted tests; Bonferroni tail unstable | Accept by removing the premise | Remove all simultaneous/multiplicity decisions and all counts of “failed cells”. Report coherent descriptive patterns metric by metric. | config/API removal; schema v3; profile/table/manuscript |
| Feature controls missing | Accept | Register EOF excursion for MMD and density, mode removal for coverage, contraction/expansion for diversity, and corresponding severity curves, using the exact training-fitted feature representation and bandwidth used by the reported TC metrics. Checkerboard artifacts additionally control high-frequency PSD and MMD, but are not imposed on local-manifold density because that response is not directionally defined. | typed control registry, amendment log, and TC control artifact |
| Uniform warming is invalid for eye contrast | Accept | Replace with separate inner-eye/eyewall perturbation. | `radial_eye_contrast_lane` |
| Phase randomisation is not a positive PSD control | Accept | Use blur/high-frequency artifacts as positive PSD controls; use 90-degree rotation as the radial Hann-PSD invariance control. Phase randomisation remains a structure control with unwindowed-power preservation recorded. | control runner and artifact |
| Controls need validated / responsive-but-unresolved / failed outcomes | Accept | Add typed control statuses; retain raw metrics but withhold affirmative interpretation after failure. Report every severity, rank response, monotonicity, first reference crossing, and invariance result rather than one PASSED label. | `ControlStatus`, `ControlRegistry`, schema-v3 cells |
| Continuous wind condition is not evaluated | Accept | Add within-class functional-versus-wind slopes with inverse storm-frequency weights, storm/sample bootstrap intervals, and synthetic responsive/ignoring validation. No equivalence verdict. | `audit_continuous_wind_response.py`; appendix JSON/CSV/figure |
| Feature findings may depend on EOF rank/fit size or high-frequency texture | Accept | Add 256/512/1024 fit-size × 10/20/40 rank audit and 16/32/64 km low-pass EOF variants. Label post hoc. | `audit_feature_robustness.py`; appendix JSON/figure |
| Structured-review search and reliability documentation | Accept; deferred to user | Preserve explicit manuscript integration points. Final submission remains blocked until searches, screening, applicability denominators, double coding, and agreement are complete. | user-owned literature-review gate |
| GenCast or other corpus coverage unclear | Accept; deferred to user | Include or justify exclusion under the reconstructed search protocol. | user-owned literature-review gate |
| DOI-bearing code/data archives required by GMD | Accept; deferred to user | Keep availability text marked as incomplete and fail the full release gate until immutable code and permission-cleared data deposits exist. | user-owned deposition gate; release audit |
| Scope too broad for one 2-D case study | Accept | Narrow v0.2 to unpaired populations of independent single-/multichannel 2-D fields. Do not claim forecasts, paired downscaling, trajectories, or coupled systems. | package/docs/manuscript scope |
| Position against CRPS, reliability, rank histograms, FSS, SAL, benchmarks | Accept | Add explicit complementary positioning: paired verification answers a different question; shared benchmarks accumulate external context whereas GAUGE creates within-study empirical context. | revised introduction/discussion and references |
| TC metric definitions insufficient | Accept | Expand definitions for geometry, masks, profiles, PSD preprocessing and units, EOF fit, MMD estimator/bandwidth, density/coverage k, diversity sign, copying regions, sampling, and generated weighting. | supplement/manuscript methods generated from registries |
| “Physical admissibility” is actually clipping contact | Accept | Rename to preprocessing saturation audit throughout. | CLI config/result labels, figures, tables, manuscript |
| C4+ pooling can hide C5 and cutoff is arbitrary | Accept | Remove C4+ from the evaluation family and all pooled tail claims. Keep Category 5 visible with 22-storm warning. | analysis protocol, config, renderer, manuscript |
| Global result confounds class composition | Accept | Retain as an as-provided dataset-composition audit only; exclude from headline profile and fidelity claims. | report provenance and paper bundle |
| Category 5 non-departures can be over-read | Accept | Show event counts, reference stability/uncertainty, unresolved status where applicable, and a dedicated C5 drill-down. Avoid agreement language. | schema-v3 cell, Fig. 5/6, appendix |
| Copying analysis has limited resolution | Accept | Report the complete copied-fraction curve and minimum boundary-crossing fraction; do not claim absence of copying below demonstrated resolution. | control artifact and memorization appendix |
| Negative gaps may reflect underdispersion or central concentration | Accept | Define gap as descriptive only and direct readers to native distributions/diversity controls; never call negative gaps “better than real”. | interpretation guide, captions, manuscript |
| PSD units unclear | Accept | Define discrepancy as mean squared log10-power difference in `dex²`; specify detrending, Hann window, energy normalization, and physical wavenumber. | TC adapter provenance, tables, manuscript |
| Fixed-bank versus new-bank/retraining uncertainty unclear | Accept | Name the central comparison range and state that it varies admissible real subsets while holding the supplied bank fixed. | schema-v3 fields and metadata |
| Degenerate q95−q50 width | Accept | Fail closed as `REFERENCE_UNRESOLVED`; validation includes a degenerate-width example. | `reference_position`; evaluation; validation |
| Figure accessibility and compressed signposts | Accept | Use distinct grayscale-safe control shapes, separate control curves in the appendix, preserve nonlinear gap axis, and keep native results/ranges visible. | presentation layer and regenerated figures |
| “Typical” example selection unclear | Accept | Keep deterministic selection from standardized seven-functional distance and export selected IDs/criterion. | renderer and example-selection table |
| Model internal name inappropriate in prose | Accept | Use “TC brightness-temperature diffusion model”; retain `baseline_ztsnr` only in provenance. | manuscript and metadata |
| Analysis described as pristine preregistration despite method development | Accept | Replace with versioned analysis protocol and amendment log; state that the archive informed development before the clean rerun. | `analysis_protocol.yaml`; manuscript |
| Need exact software version in title/metadata | Accept | Release breaking changes as `gauge-climate` v0.2.0 and report schema v3; use version in GMD title. | package metadata, changelog, manuscript |
| Sampler/model training details and 250 steps need confirmation | Accept | Report configuration directly from the frozen model metadata and distinguish case-model reproducibility from GAUGE's method claims. | manuscript model subsection and provenance audit |
| Literature paper is methods contribution, not a TC-physics result | Accept | Keep TC as a demanding demonstration; frame conclusions around evaluation behaviour and revealed evidence limitations. | abstract, discussion, conclusions |

## Readiness rule

An item is complete only when the revised statement is generated from, or
traceable to, the cited artifact. The literature-review and permanent-deposition
rows remain intentionally open and block final submission even if all local
software and manuscript checks pass.
