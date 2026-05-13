# Tropical Cyclone Brightness Temperature Case Study

This case study evaluates generated tropical-cyclone brightness-temperature images
against the held-out test split from the original `tc-diffusion` project.

Prepared local data live under:

```text
data/tropical_cyclone_bt/
  real_test/
    bt_k.npy
    labels.npy
    wind_kt.npy
    rel_paths.txt
    manifest.json
  generated/
    FINAL_BASELINE/final_bank -> source sample bank
    pinn_loss/final_bank      -> source sample bank
  source_manifests/
```

The generated banks are symlinked by default so that this repo stays lightweight.
Use `--copy-mode copy` in `prepare_case_study.py` when you need a physically
self-contained local archive.

The `FINAL_BASELINE` bank used here is:

```text
outputs/FINAL_BASELINE/sample_banks/test/final_bank
```

That choice is inferred from timestamps: `final_eval_1000` was written on
2026-04-30, while the later reranked sample bank was created on 2026-05-01.

