# Metric selection guide

Choose metrics from the scientific claims, not from a universal checklist.

| Claim | Register | Report | Common failure |
|---|---|---|---|
| A scalar physical quantity has the right distribution | `FunctionalSpec` + W1 | native-unit W1, direct interval, tolerance, reference envelope | choosing the functional/tolerance after viewing test results |
| Generated modes cover the real feature manifold | train-fit embedding + density/coverage | both density departure and coverage deficit | fitting PCA, scale, or bandwidth on test/generated data |
| Dependence among physical summaries is preserved | multivariate functional transform + Spearman closure | matrix discrepancy and interpretable pairwise diagnostics | claiming multivariate fidelity from marginal W1 alone |
| Spatial/scale structure is preserved | domain adapter curve metric | declared coordinates, preprocessing, masks, and curve distance | treating a radial PSD as universal |
| Samples are admissible | hard/soft domain predicate | raw generated rate and reference-calibrated rate departure | hiding invalid samples before evaluation |
| The model is not copying training samples | train-fit feature space + localized copying test | label-matched region effects, simultaneous intervals, copyist control | only comparing global nearest-neighbor histograms or confounding query-label mixtures |
| Samples improve a downstream task | eight-arm matched ladder | paired generated-vs-copy contrasts and predeclared task margin | comparing augmentation only with no augmentation |

W1 is a common currency only for scalar, physically interpretable functionals.
It is not a replacement for conditional, spatial, temporal, coupled-variable, or
downstream questions. Non-functional metrics receive a detection result but no
invented native-unit equivalence conclusion unless a defensible tolerance and
validated inference method are registered.
