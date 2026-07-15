# Troubleshooting

- **Train/test block leakage:** all frames from a physical event must remain on
  one side of the split. For TC data, the block is the storm, not the frame or
  intensity label.
- **Thin conditional stratum:** report it as underpowered and use only a
  scientifically predeclared aggregate (for example C4+)—never merge after
  inspecting model results.
- **Functional outside support:** correct or justify the physical support before
  running. GAUGE refuses to clip values silently.
- **No practical verdict:** v0.1 finite-sample W1 tolerance inference needs a
  bounded scalar functional and the block-balanced estimand. The reference
  result remains usable for detection; clustered bootstrap output is diagnostic.
- **Lazy xarray exceeds memory:** set `max_materialize_bytes`, rechunk upstream,
  or evaluate smaller predeclared batches. GAUGE never triggers uncontrolled
  materialization deliberately.
- **Positive control fails:** do not interpret that metric. Revisit preprocessing,
  severity, feature fitting, and valid input domain.
- **Different labels in real/generated data:** conditional support must match.
  Missing classes are a substantive coverage failure that should be handled by a
  predeclared unconditional or explicit missing-class analysis, not silently.
