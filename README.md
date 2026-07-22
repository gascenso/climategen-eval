# GAUGE Climate

GAUGE is a reference-aware evaluation protocol for independently generated
Earth-system fields. It combines physical functionals, distributional metrics,
finite-sample real-reference calibration, sensitivity controls, diversity,
coverage, and memorization checks. It deliberately does **not** reduce a model
to one aggregate score.

The import namespace is `gauge_climate`; the distribution and command are
`gauge-climate`. Version 0.1 targets NumPy/memmap arrays and independent 2-D
single- or multichannel samples. Arbitrary coupled and spatiotemporal systems
are extension points, not claimed as solved.

## Install

```bash
python -m pip install -e .
```

Use `.[xarray]` for the optional xarray/Dask adapter and `.[dev]` to run the
test suite.

## One-screen example

```python
import numpy as np
import gauge_climate as gauge

rng = np.random.default_rng(7)
train = gauge.Dataset(
    rng.normal(size=(80, 16, 16)), blocks=np.repeat([f"tr{i}" for i in range(40)], 2)
)
real = gauge.Dataset(
    rng.normal(size=(80, 16, 16)), blocks=np.repeat([f"te{i}" for i in range(40)], 2)
)
generated = gauge.Dataset(rng.normal(size=(80, 16, 16)))

mean = gauge.FunctionalSpec(
    id="domain_mean",
    function=lambda x: x.mean(axis=(1, 2)),
    units="standardized units",
    description="domain-mean field value",
)
report = gauge.evaluate(
    train=train,
    real=real,
    generated=generated,
    functionals=[mean],
    config=gauge.EvaluationConfig(n_eval=80, n_resamples=200),
)
report.to_json("evaluation.json")
report.to_html("evaluation.html")
```

The zero-configuration report is reference-aware:

- every scalar functional is reported as a W1 discrepancy in its native units;
- the real-reference envelope shows ordinary real-vs-real variation at the same
  declared sample size and block-weighting policy;
- the reference gap places the model relative to that envelope; and
- sensitivity controls show whether each registered metric responds to known
  degradations and copying.

GAUGE does not ask users to invent application thresholds and does not issue a
universal equivalence verdict. Corruption controls validate metric response and
provide interpretable failure anchors; they do not define what is acceptable
for every application.

## Paper-facing output

The complete statistical report is designed to sit underneath a much smaller
results package. Conditional applications can render an interpretation-first
profile, exact native-unit table, complete class matrix, and browser landing
page directly from the frozen JSON:

```python
spec = gauge.PaperBundleSpec(
    title="Generated field evaluation",
    model_label="primary model",
    metrics=("w1:domain_mean",),
    native_strata=("low", "middle", "high"),
    priority_strata=("upper_tail",),
    metric_labels={"w1:domain_mean": "Domain mean · W1"},
)
report.paper_bundle("paper_results", spec)
```

When there are at most eight native strata, the primary profile displays every
one. With larger condition sets it displays the native-stratum median and range,
labels the largest observed gap, and retains every preregistered priority stratum.
The appendix matrix always contains every class. The reference gap is a compact
detectability coordinate—0 is the real-vs-real median and 1 the pointwise 95%
boundary—not an aggregate score or a substitute for native-unit inference.

## Command line

```bash
gauge-climate validate path/to/config.yaml
gauge-climate run path/to/config.yaml
gauge-climate report evaluation.json --html evaluation.html --profile profile.pdf
```

Start with the [preregistration template](docs/templates/gauge_preregistration.yaml),
[metric-selection guide](docs/metric_selection.md), and
[interpretation guide](docs/interpretation.md). The
[statistical contract](docs/statistical_contract.md) states the assumptions and
finite-sample limitations. The scientific contract and its
statistical qualifications are in [the blueprint](docs/methods/gauge_blueprint.md).
The tropical-cyclone case study is tracked separately under
[`case_studies/tropical_cyclone_bt`](case_studies/tropical_cyclone_bt).
Its preregistered training-only controls run with
`python case_studies/tropical_cyclone_bt/run_track1_controls.py`. The frozen
machine report is in
[`final_results`](case_studies/tropical_cyclone_bt/final_results); the concise
results-section and appendix bundle an ordinary GAUGE user would report is in
[`paper_results`](case_studies/tropical_cyclone_bt/paper_results).

For conditional generators, GAUGE treats the condition distribution as part of
the estimand. If real and generated label proportions differ, global rows are
explicitly labeled as supplied-dataset composition checks; conditional-fidelity
claims must come from matched, preregistered strata or a declared mixture rule.

## Development state

This repository is pre-1.0. The validated reference-aware engine and public API
are the v0.1 critical path; survey reconciliation, clean-room TC reproduction,
the user study, and independent statistical review remain explicit release
gates. See [CHANGELOG.md](CHANGELOG.md).
The exact implemented-versus-pending matrix is in
[docs/development_status.md](docs/development_status.md).

## License and citation

Code is BSD-3-Clause. See [CITATION.cff](CITATION.cff) for provisional citation
metadata; the archival DOI will be added at the paper freeze.
