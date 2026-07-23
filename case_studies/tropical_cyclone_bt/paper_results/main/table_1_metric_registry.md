# Table 1. Registered TC evaluation metrics

| Metric | Family | Units | Definition | Control / provenance |
|---|---|---|---|---|
| Core minimum BT · W1 | functional_w1 | K | Wasserstein-1 distance for minimum brightness temperature within 100 km | uniform BT shift |
| Cold-cloud area · W1 | functional_w1 | % | Wasserstein-1 distance for area fraction below 220 K within 300 km | uniform BT shift |
| Max-gradient radius · W1 | functional_w1 | km | Wasserstein-1 distance for radius of the largest absolute azimuthal-mean BT gradient | blur and center jitter |
| Axisymmetry · W1 | functional_w1 | K | Wasserstein-1 distance for RMS azimuthal departure within 300 km | phase shuffle |
| Deviation-angle variance · W1 | functional_w1 | deg² | Wasserstein-1 distance for deviation-angle variance within 300 km | phase shuffle |
| Eye contrast · W1 | functional_w1 | K | Wasserstein-1 distance for mean eye brightness temperature minus mean eyewall brightness temperature | uniform BT shift |
| BT lower tail · W1 | functional_w1 | K | Wasserstein-1 distance for per-image first percentile of brightness temperature | uniform BT shift |
| Radial BT profile · MAE | curve | K | MAE between mean azimuthal BT profiles on fixed physical radial bins. | center jitter and phase shuffle |
| Radial power spectrum · MSE | curve | log10-PSD² | MSE between mean Hann-windowed radial log10 power spectra. | blur and phase-power preservation |
| Feature distribution · MMD² | set | MMD² | Unbiased Gaussian-kernel MMD² in a training-fitted feature space. | training-fitted feature audit |
| Feature density departure | set | absolute density departure | Absolute departure from Naeem-style feature-space density=1. | training-fitted feature audit |
| Feature coverage deficit | set | fraction | Naeem-style real-manifold coverage deficit (1 - coverage). | training-fitted feature audit |
| Feature diversity departure | set | relative departure | Absolute relative departure in mean feature-space pairwise distance. | training-fitted feature audit |
