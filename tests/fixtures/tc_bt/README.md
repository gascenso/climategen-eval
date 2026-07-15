# Synthetic TC-BT fixture

This directory contains a deterministic, fully synthetic 32×32 storm-like
fixture for adapter and CLI smoke tests. It contains no observations and is not
scientific evidence for the paper. `manifest.json` records every array checksum;
regenerate it with `python tests/fixtures/tc_bt/generate_fixture.py`.

Its integer labels are synthetic software-test strata. They are not derived
from IBTrACS and do not encode either WMO_WIND or USA_WIND.

```bash
gauge-climate validate tests/fixtures/tc_bt/config.yaml
gauge-climate run tests/fixtures/tc_bt/config.yaml
```
