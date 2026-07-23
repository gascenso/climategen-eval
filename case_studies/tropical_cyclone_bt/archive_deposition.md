# TC artifact deposition handoff

The local datasets are complete and checksum-audited. They remain outside Git
because their logical size is approximately 18.9 GB. Run

```bash
python scripts/build_submission_inventory.py
```

immediately before deposition. The resulting `submission_inventory.json`
records every local dataset path, logical byte count, checksum-manifest digest,
and frozen evidence digest.

Deposit the following four directories without renaming files:

1. `data/tropical_cyclone_bt/real_train`
2. `data/tropical_cyclone_bt/real_val`
3. `data/tropical_cyclone_bt/real_test`
4. `data/tropical_cyclone_bt/generated/baseline_ztsnr/test/test_baseline_n1000`

The first three archives must preserve the `USA_WIND`/one-minute label
provenance in their manifests. The generated-bank checksum contract is
`data/tropical_cyclone_bt/generated/manifests/baseline_ztsnr_test.json`.

After the repository or institutional data steward confirms redistribution
permission:

1. create immutable archive records and obtain persistent identifiers;
2. verify downloaded files against the included manifests;
3. replace only the corresponding `PENDING` values in `artifacts.yaml`;
4. set `release_ready: true` only after a clean download passes
   `python scripts/audit_release_gates.py`;
5. add the software/data DOI to `CITATION.cff` and the manuscript data
   availability statement.

Do not substitute links into the sibling `tc-diffusion` checkout: a local path
is provenance, not an archive.
