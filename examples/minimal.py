"""Small, bounded synthetic GAUGE evaluation suitable for a clean install."""

from pathlib import Path

import numpy as np

import gauge_climate as gauge


def main() -> None:
    rng = np.random.default_rng(12)
    train_fields = rng.uniform(-1, 1, size=(80, 8, 8))
    real_fields = rng.uniform(-1, 1, size=(80, 8, 8))
    generated_fields = rng.uniform(-1, 1, size=(80, 8, 8))
    train = gauge.Dataset(train_fields, blocks=np.repeat(np.arange(40), 2))
    real = gauge.Dataset(real_fields, blocks=np.repeat(np.arange(100, 140), 2))
    generated = gauge.Dataset(generated_fields)
    functional = gauge.FunctionalSpec(
        id="mean",
        function=lambda fields: fields.mean(axis=(1, 2)),
        units="field units",
        description="spatial mean",
        support=(-1.0, 1.0),
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=(functional,),
        config=gauge.EvaluationConfig(n_resamples=100),
    )
    output = Path("outputs/minimal")
    report.to_json(output / "evaluation.json")
    report.to_html(output / "evaluation.html")


if __name__ == "__main__":
    main()
