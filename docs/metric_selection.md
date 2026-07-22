# Metric selection guide

Choose metrics from the scientific claims, not from a universal checklist.

| Claim | Register | Report | Common failure |
|---|---|---|---|
| A scalar physical quantity has the right distribution | `FunctionalSpec` + W1 | native-unit W1, resampling interval, reference envelope, and gap | choosing the functional after viewing test results |
| Generated modes cover the real feature manifold | train-fit embedding + density/coverage | both density departure and coverage deficit | fitting PCA, scale, or bandwidth on test/generated data |
| Dependence among physical summaries is preserved | multivariate functional transform + Spearman closure | matrix discrepancy and interpretable pairwise diagnostics | claiming multivariate fidelity from marginal W1 alone |
| Spatial/scale structure is preserved | domain adapter curve metric | declared coordinates, preprocessing, masks, and curve distance | treating a radial PSD as universal |
| Samples are admissible | hard/soft domain predicate | raw generated rate and reference-calibrated rate departure | hiding invalid samples before evaluation |
| The model is not copying training samples | train-fit feature space + localized copying test | label-matched region effects, simultaneous intervals, copyist control | only comparing global nearest-neighbor histograms or confounding query-label mixtures |

W1 is a common currency only for scalar, physically interpretable functionals.
It is not a replacement for conditional, spatial, temporal, or coupled-variable
questions. GAUGE makes no universal equivalence claim: the real-reference gap
and controlled degradations contextualize detectability, while native-unit
values preserve the physical evidence.
