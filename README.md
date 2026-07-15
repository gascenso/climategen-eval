# GAUGE Climate

GAUGE is a reference-aware evaluation protocol for independently generated
Earth-system fields. It combines physical functionals, distributional metrics,
finite-sample real-reference calibration, direct practical tolerances, controls,
and leakage-audited downstream studies. It deliberately does **not** reduce a
model to one aggregate score.

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
    tolerance=0.25,              # frozen before final evaluation
    support=(-5.0, 5.0),         # scientifically defensible finite support
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

The report has two separate conclusions:

- `DEPARTURE_DETECTED` means the model discrepancy exceeded the declared
  perfect-generator reference boundary at this sample size. Its complement is
  `NO_DEPARTURE_DETECTED`, never “equal” or “indistinguishable.”
- `WITHIN_TOLERANCE` means a direct upper confidence bound for population W1 is
  below the predeclared physical tolerance. Other outcomes are
  `EXCEEDS_TOLERANCE`, `INCONCLUSIVE`, or `UNDERPOWERED`.

Finite-sample tolerance inference in v0.1 requires the block-balanced estimand
and a declared finite support for the scalar functional. Otherwise GAUGE shows
the reference result and a clustered-bootstrap diagnostic but withholds an
equivalence claim.

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

## Development state

This repository is pre-1.0. The validated Track-1 engine and public API are the
v0.1 critical path; survey reconciliation, clean-room TC reproduction, three
restricted generator retrains, user study, and independent statistical review
remain explicit release gates. See [CHANGELOG.md](CHANGELOG.md).
The exact implemented-versus-pending matrix is in
[docs/development_status.md](docs/development_status.md).

## License and citation

Code is BSD-3-Clause. See [CITATION.cff](CITATION.cff) for provisional citation
metadata; the archival DOI will be added at the paper freeze.
