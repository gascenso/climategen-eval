# Troubleshooting

- **Train/test event leakage:** all fields from one physical event must remain on
  one side of the split. For the TC adapter, the event block is the storm, not a
  frame or intensity category.
- **Missing or wrong block identifiers:** GAUGE cannot infer the scientifically
  independent unit. Supply storm, day, ensemble-member, or other event IDs that
  match the sampling process. The validation examples show how frame-IID IDs
  produce misleadingly narrow references.
- **Functional outside its declared support:** correct the implementation or
  justify the support. GAUGE does not silently clip generic inputs.
- **`NOT_EVALUATED` for a thin condition:** report the field and event counts.
  Do not merge conditions after inspecting outcomes.
- **`REFERENCE_UNRESOLVED`:** increase the real-reference repetition budget if
  feasible. If few independent events are the limiting factor, retain the
  warning; more resampling cannot create information absent from the archive.
- **Lazy archive exceeds memory:** lower `transform_batch_size`. For
  training-fitted EOF features, lower the declared fit sample count or feature
  batch size if scientifically acceptable and record the change.
- **Secondary metrics are slow:** use a declared matched sample cap for
  quadratic feature-set metrics. Do not change it after viewing model results
  without recording an analysis amendment.
- **Control fails:** keep the raw metric visible but do not make an affirmative
  interpretation from it. Revisit preprocessing, control construction, and the
  metric's valid input domain.
- **Control responds but remains below the reference range:** report
  `RESPONSIVE_BELOW_REFERENCE_RANGE`. The metric has the expected ordering but
  lacks resolution for the registered severity range at this design.
- **Real and generated labels differ:** native conditional support must match.
  The as-provided global row is a composition audit when label proportions
  differ.
