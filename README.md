# GAUGE Climate

GAUGE is an empirical reference-calibration protocol for unpaired populations
of independently generated Earth-system fields. It combines physical
functionals, distributional metrics, finite-archive real-reference calibration,
sensitivity and invariance controls, diversity, coverage, and memorization
checks. It deliberately does **not** reduce a model to one aggregate score.

The import namespace is `gauge_climate`; the distribution and command are
`gauge-climate`. Version 0.2 targets NumPy/memmap arrays and independent 2-D
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

The standard report is reference-aware:

- every scalar functional is reported as a W1 discrepancy in its native units;
- the real-reference distribution shows ordinary real-vs-real variation at the same
  declared sample size and block-weighting policy;
- the reference percentile and gap place the model relative to that distribution; and
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
labels the largest observed gap, and retains every registered priority stratum.
The appendix matrix always contains every class. The reference gap is a compact
finite-archive coordinate—0 is the median discrepancy between event-disjoint
real subsets and 1 their 95th-percentile marker—not a p-value, aggregate score,
physical severity scale, or substitute for native-unit evidence.

## Command line

```bash
gauge-climate validate path/to/config.yaml
gauge-climate run path/to/config.yaml
gauge-climate report evaluation.json --html evaluation.html --profile profile.pdf
```

Start with the [analysis-protocol template](docs/templates/gauge_analysis_protocol.yaml),
[metric-selection guide](docs/metric_selection.md), and
[interpretation guide](docs/interpretation.md). The
[statistical contract](docs/statistical_contract.md) states the assumptions and
finite-sample limitations. The scientific contract and its
statistical qualifications are in [the blueprint](docs/methods/gauge_blueprint.md).
The tropical-cyclone case study is tracked separately under
[`case_studies/tropical_cyclone_bt`](case_studies/tropical_cyclone_bt).
Its versioned training-only controls run with
`python case_studies/tropical_cyclone_bt/run_track1_controls.py`. The frozen
machine report and complete HTML evaluation card are in
[`final_results`](case_studies/tropical_cyclone_bt/final_results).

For conditional generators, GAUGE treats the condition distribution as part of
the estimand. If real and generated label proportions differ, global rows are
explicitly labeled as supplied-dataset composition checks; conditional-fidelity
claims must come from matched, registered strata or a declared mixture rule.

## Development state

This repository is pre-1.0. The empirical-calibration engine and TC evidence
have been regenerated and frozen for v0.2. Independent
statistical review, permission-cleared data deposition, and clean-room TC
reproduction remain archival release gates. See [CHANGELOG.md](CHANGELOG.md).
The exact implemented-versus-pending matrix is in
[docs/development_status.md](docs/development_status.md).

## License and citation

Code is BSD-3-Clause. See [CITATION.cff](CITATION.cff) for provisional citation
metadata; the archival DOI will be added at the paper freeze.
