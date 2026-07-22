# Troubleshooting

- **Train/test block leakage:** all frames from a physical event must remain on
  one side of the split. For TC data, the block is the storm, not the frame or
  intensity label.
- **Functional outside support:** correct or justify the physical support before
  running. GAUGE refuses to clip values silently.
- **No detection result for a thin stratum:** report the block and sample counts,
  retain the native class in the appendix, and use only a scientifically
  predeclared aggregate—never merge after inspecting model outcomes.
- **Lazy archive exceeds memory:** lower `transform_batch_size`; for train-fit
  EOF features, also lower the preregistered `max_fit_samples` or feature-space
  batch size. `max_materialize_bytes` guards explicit `Dataset.as_numpy()` calls;
  the evaluation engine and TC sample-bank adapter process fields in batches.
- **Secondary metrics are slow:** freeze a smaller `diagnostic_resamples` budget
  and, for quadratic feature-set metrics, a matched `max_samples_per_draw` cap.
  Do not reduce `n_resamples` for functional W1 or a registered simultaneous
  family after seeing outcomes; all effective counts must remain in provenance.
- **Positive control fails:** do not interpret that metric. Revisit preprocessing,
  severity, feature fitting, and valid input domain.
- **Different labels in real/generated data:** conditional support must match.
  Missing classes are a substantive coverage failure that should be handled by a
  predeclared unconditional or explicit missing-class analysis, not silently.
