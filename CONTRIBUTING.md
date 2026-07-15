# Contributing

GAUGE changes are reviewed against the scientific contract as well as software
quality. Before opening a change:

1. Add or update a preregistration field for any new metric, tolerance,
   perturbation, embedding, or endpoint.
2. Declare the estimand, valid input domain, orientation, units, provenance,
   expected positive-control response, and permitted interpretation.
3. Add equality, failure, and predeclared positive-control tests. Statistical
   changes also require Monte Carlo coverage/type-I-error evidence.
4. Run `python -m pytest` and `python scripts/validate_gauge_statistics.py`.
5. Do not use final test results to choose a tolerance, endpoint, stratum merge,
   or perturbation severity.

New domain functionals belong in an adapter unless they are scientifically
meaningful for every supported field type. Do not add an aggregate model score.
