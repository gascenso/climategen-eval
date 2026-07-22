# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
semantic versioning once the public API is frozen.

## [Unreleased]

- Independent statistical review and clean-room TC reproduction.
- Independent review of the dependence-matched real-reference detector before
  the final evaluation.
- Reconciled and independently double-coded literature survey.

### Added

- Frozen, training-only TC warm-shift, blur, center-jitter, phase, and exact-copyist
  controls with a versioned JSON artifact and figure.
- Separate sensitivity response-contract and finite-sample resolution conclusions.
- Versioned 500-trial Monte Carlo validation artifact for the generic
  dependence-matched real-reference detector.

### Changed

- TC radial PSD bins now use discrete-support-aware physical log targets, avoiding
  undefined empty low-wavenumber shells on the registered 256×256 grid.
- TC analysis now applies the generator-training 170--325 K clipping lazily to
  every analysis input while preserving raw archive bytes and checksums.
- Shift-resolution evaluation now limits both the reference draws and bootstrap
  work, eliminating an unintended full-resample computation at every grid point.

## [0.1.0] - 2026-07-15

### Added

- Portable `gauge_climate` evaluation API and `gauge-climate` CLI.
- Matched, dependence-aware real-reference engine with explicit weighting estimands.
- Calibrate detection against dependence-matched real-reference envelopes.
- Matched model-versus-reference resampling intervals for raw discrepancies.
- Explicitly preregistered Bonferroni simultaneous families with finite-Monte-Carlo
  resolution checks.
- Train-fit feature spaces, diversity/coverage/dependence metrics, localized
  copying test, sensitivity lanes, versioned JSON/HTML reporting, and TC adapter.
- Checksummed synthetic TC fixture and fail-closed survey/artifact/release audits.
