# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
semantic versioning once the public API is frozen.

## [Unreleased]

- Independent statistical review and clean-room TC reproduction.
- TC-specific statistical calibration before the final Track-1 run.
- Three leakage-free restricted generator retrains and the final Track-2 study.
- Reconciled and independently double-coded literature survey.

### Added

- Frozen, training-only TC warm-shift, blur, center-jitter, phase, and exact-copyist
  controls with a versioned JSON artifact and figure.
- Separate sensitivity response-contract and finite-sample resolution conclusions.

### Changed

- TC radial PSD bins now use discrete-support-aware physical log targets, avoiding
  undefined empty low-wavenumber shells on the registered 256×256 grid.

## [0.1.0] - 2026-07-15

### Added

- Portable `gauge_climate` Track-1 API and `gauge-climate` CLI.
- Matched, dependence-aware real-reference engine with explicit weighting estimands.
- Separate detection and practical-tolerance conclusions.
- Finite-sample DKW quantile-band intervals for bounded scalar functional W1.
- Explicitly preregistered Bonferroni simultaneous families with finite-Monte-Carlo
  resolution checks.
- Train-fit feature spaces, diversity/coverage/dependence metrics, localized
  copying test, sensitivity lanes, versioned JSON/HTML reporting, and TC adapter.
- Leakage-audited eight-arm generic downstream ladder.
- Checksummed synthetic TC fixture and fail-closed survey/artifact/release audits.
