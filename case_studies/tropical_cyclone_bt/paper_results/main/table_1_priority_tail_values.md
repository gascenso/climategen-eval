# C4+ priority-tail metric values

| Metric | Model discrepancy (95% CI) | Real-reference median → boundary | Gap | Detection | Practical |
|---|---:|---:|---:|---|---|
| Core minimum BT · W1 | 0.624 [0, 1.36] K | 0.925 → 1.94 | -0.15 | NO_DEPARTURE_DETECTED | INCONCLUSIVE |
| Cold-cloud area · W1 | 1.92 [0.114, 3.72] % | 2.91 → 6.45 | -0.11 | NO_DEPARTURE_DETECTED | UNDERPOWERED |
| Max-gradient radius · W1 | 11.2 [0, 117] km | 20.1 → 45.5 | -0.24 | NO_DEPARTURE_DETECTED | UNDERPOWERED |
| Axisymmetry · W1 | 0.251 [0, 0.569] K | 0.535 → 1.02 | -0.35 | NO_DEPARTURE_DETECTED | WITHIN_TOLERANCE |
| Deviation-angle variance · W1 | 28.5 [0, 62.3] deg² | 65.9 → 134 | -0.31 | NO_DEPARTURE_DETECTED | WITHIN_TOLERANCE |
| Eye contrast · W1 | 0.799 [0.15, 1.45] K | 1.85 → 3.76 | -0.33 | NO_DEPARTURE_DETECTED | INCONCLUSIVE |
| BT lower tail · W1 | 0.879 [0.315, 1.44] K | 0.975 → 2.04 | 0.04 | NO_DEPARTURE_DETECTED | INCONCLUSIVE |
| Radial BT profile · MAE | 0.866 [0.341, 2.19] K | 1.69 → 3.46 | -0.46 | NO_DEPARTURE_DETECTED | NOT_APPLICABLE |
| Radial power spectrum · MSE | 0.00422 [0.00269, 0.00663] log10-PSD² | 0.00143 → 0.00373 | 1.21 | DEPARTURE_DETECTED | NOT_APPLICABLE |
| Feature distribution · MMD² | 0.00791 [0.00472, 0.0119] MMD² | 0.00942 → 0.0154 | -0.25 | NO_DEPARTURE_DETECTED | NOT_APPLICABLE |
| Feature density departure | 0.631 [0.516, 0.713] absolute density departure | 0.421 → 0.577 | 1.34 | DEPARTURE_DETECTED | NOT_APPLICABLE |
| Feature coverage deficit | 0.391 [0.305, 0.488] fraction | 0.453 → 0.539 | -0.73 | NO_DEPARTURE_DETECTED | NOT_APPLICABLE |
| Feature diversity departure | 0.0804 [0.0471, 0.116] relative departure | 0.0217 → 0.061 | 1.49 | DEPARTURE_DETECTED | NOT_APPLICABLE |

Gap 1 is the pointwise 95% real-reference boundary. Non-functional diagnostics have no fabricated practical-equivalence status.
