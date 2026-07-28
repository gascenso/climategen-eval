# Statistical validation artifacts

Regenerate the v0.2 validation report and paper-ready PDF/PNG with:

```bash
MPLCONFIGDIR=/tmp/mpl-gauge PYTHONPATH=src \
  python scripts/validate_gauge_statistics.py \
  --trials 300 --resamples 200 --jobs 4
```

The command writes the schema-v4 machine-readable report, the four-panel
main-text figure (`statistical_validation_v0.2.{pdf,png}`), and a separate
stability/finite-archive figure
(`statistical_validation_v0.2_appendix.{pdf,png}`).

The harness uses neutral empirical-calibration quantities:

- **same-population boundary-crossing rate** rather than false-positive rate;
- **perturbation crossing probability** rather than statistical power.

It covers:

- event-disjoint plan and fixed-sample-size invariants;
- Gaussian, heavy-tailed, multimodal, shifted, narrowed, widened, crossing, and
  tail-contaminated scalar populations;
- curve MAE and log-PSD discrepancy under known spatial changes;
- non-negative MMD², density, coverage, and diversity under feature-space
  degenerations;
- balanced, severely unequal, 22-event, and minimum-valid 20-event designs;
- correct event identifiers versus intentionally wrong frame-IID identifiers;
- multiple bank sizes and train/test population shifts;
- reference-quantile and gap Monte Carlo stability;
- grouped delete-event finite-archive sensitivity;
- fail-closed behaviour for insufficient events or degenerate reference widths.

Acceptance requires same-population crossing rates not to exceed the nominal 5%
reference tail beyond Monte Carlo uncertainty, increasing crossing for
increasing perturbations, expected monotonic control responses, visibly lower
resolution with few events, and non-negative biased/V-statistic MMD² up to
numerical tolerance.

These simulations validate the behaviour of the empirical reference
calculation. They do not turn the reference quantile into an exact
hypothesis-test threshold.
