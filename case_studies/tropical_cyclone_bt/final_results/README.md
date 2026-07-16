# Frozen TC Track-1 results

Source report SHA-256: `ae7089d063891ac7e812e6035b66ecc763e9c0fbef9f37e9cdf0f3d670e01b00`; config hash: `9d4bdd409257157f`.

## Primary scalar-function results

Across all 56 registered cells, pointwise detection counts are `{'DEPARTURE_DETECTED': 7, 'NO_DEPARTURE_DETECTED': 49}` and simultaneous counts are `{'NO_DEPARTURE_DETECTED': 52, 'DEPARTURE_DETECTED': 4}`. Practical statuses are `{'EXCEEDS_TOLERANCE': 7, 'UNDERPOWERED': 23, 'INCONCLUSIVE': 13, 'WITHIN_TOLERANCE': 13}`.

Across the 42 exact class-conditioned cells (TS and C1–C5), detection counts are `{'NO_DEPARTURE_DETECTED': 42}` and practical statuses are `{'INCONCLUSIVE': 10, 'UNDERPOWERED': 20, 'WITHIN_TOLERANCE': 11, 'EXCEEDS_TOLERANCE': 1}`. The seven predeclared C4+ cells have `{'NO_DEPARTURE_DETECTED': 7}` with practical statuses `{'INCONCLUSIVE': 3, 'UNDERPOWERED': 2, 'WITHIN_TOLERANCE': 2}`.

The seven global as-provided-mixture cells have pointwise counts `{'DEPARTURE_DETECTED': 7}` and simultaneous counts `{'NO_DEPARTURE_DETECTED': 3, 'DEPARTURE_DETECTED': 4}`. These rows compare a balanced generated bank with the natural test label mixture and are a composition check, not conditional-generator fidelity.

## Memorization

Across 41 supported query cells, 0 report copying, 5 are significantly farther from training than held-out real queries, and 36 are inconclusive. The legacy aggregate machine enum is `OFF_MANIFOLD`; here it means farther-from-training and does not establish physical inadmissibility.

## Monte Carlo audit

Primary functional cells with stability warnings: 0. Secondary cells with warnings: 44, by kind `{'curve': 4, 'matrix': 35, 'set': 3, 'rate': 2}`. A dagger in secondary figures marks an unstable 200-draw diagnostic boundary; those detection labels are not headline claims.

## Figure key

- `track1_functional_card.pdf`: practical-status heatmap with pointwise/simultaneous detection overlay.
- `track1_reference_profile.pdf`: compact detectability profile for the 56 scalar-functional cells.
- `track1_radial_profiles.pdf` and `track1_radial_psd.pdf`: stored curve diagnostics and real-reference envelopes.
- `track1_feature_diagnostics.pdf`: MMD², density, coverage, and diversity detection map.
- `track1_dependence.pdf`: raw pairwise Spearman-correlation gaps; daggers mark unstable categorical boundaries.
- `track1_admissibility.pdf`: real/generated preprocessing-boundary saturation rates.
- `track1_memorization.pdf`: localized block-aware three-sample copying result.
