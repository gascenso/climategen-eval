# Metric selection guide

Choose metrics from the scientific claims, not from a universal checklist.

| Claim | Register | Report | Common failure |
|---|---|---|---|
| A scalar physical quantity has the right distribution | `FunctionalSpec` + W1 | native-unit W1, central comparison range, empirical real reference, and gap | choosing the functional after viewing test results |
| Generated modes cover the real feature manifold | train-fit embedding + density/coverage | both density departure and coverage deficit | fitting PCA, scale, or bandwidth on test/generated data |
| Dependence among physical summaries is preserved | multivariate functional transform + Spearman closure | matrix discrepancy and interpretable pairwise diagnostics | claiming multivariate fidelity from marginal W1 alone |
| Spatial/scale structure is preserved | domain adapter curve metric | declared coordinates, preprocessing, masks, and curve distance | treating a radial PSD as universal |
| Preprocessing creates boundary artifacts | registered boundary predicate | raw boundary-contact rate and reference-calibrated departure | calling a preprocessing audit proof of physical admissibility |
| The model does not replay training samples | train-fit feature space + localized copying audit | label-matched regional effects and copy-fraction resolution | only comparing global nearest-neighbor histograms or confounding query-label mixtures |

W1 is a common currency only for scalar, physically interpretable functionals.
It is not a replacement for conditional, spatial, temporal, or coupled-variable
questions. GAUGE makes no universal equivalence claim: the empirical
real-reference gap and controlled degradations provide finite-archive context,
while native-unit values preserve the physical evidence.
