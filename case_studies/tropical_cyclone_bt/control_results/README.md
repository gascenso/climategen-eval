# Frozen Track-1 controls

`sensitivity_lanes.json` and `sensitivity_lanes.pdf` were generated exclusively
from the corrected training archive with:

```bash
PYTHONPATH=src python case_studies/tropical_cyclone_bt/run_track1_controls.py
```

The JSON artifact records the exact input and runner hashes, selected training
blocks, real-reference summaries, response-contract results, resolution at the
maximum registered severity, and the localized copyist result. No validation,
test, or generated-model fields enter these controls.

Passing a sweep means the registered metric has the required severity trend.
Crossing the real-reference boundary is reported separately as finite-sample
resolution. Full phase shuffling and the exact-copyist lane are binary positive
controls and must be detected to pass.
