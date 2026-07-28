# Table 1. Registered TC evaluation metrics

| Metric | Family | Units | Definition | Control / provenance |
|---|---|---|---|---|
| Core minimum BT · W1 | functional_w1 | K | Wasserstein-1 distance for minimum brightness temperature within 100 km | additive temperature shift |
| Cold-cloud area · W1 | functional_w1 | % | Wasserstein-1 distance for area fraction below 220 K within 300 km | warming near the 220 K threshold |
| Max-gradient radius · W1 | functional_w1 | km | Wasserstein-1 distance for radius of the largest absolute azimuthal-mean BT gradient | blur, centre displacement, and radial deformation |
| Axisymmetry · W1 | functional_w1 | K | Wasserstein-1 distance for RMS azimuthal departure within 300 km | Fourier-phase randomisation |
| Deviation-angle variance · W1 | functional_w1 | deg² | Wasserstein-1 distance for deviation-angle variance within 300 km | Fourier-phase randomisation |
| Eye contrast · W1 | functional_w1 | K | Wasserstein-1 distance for mean eye brightness temperature minus mean eyewall brightness temperature | eye/eyewall temperature perturbation |
| BT lower tail · W1 | functional_w1 | K | Wasserstein-1 distance for per-image first percentile of brightness temperature | additive temperature shift |
| Radial BT profile · MAE | curve | K | MAE between mean azimuthal BT profiles on fixed physical radial bins. | centre displacement, radial deformation, and phase randomisation |
| Radial power spectrum · MSE | curve | dex² | Mean squared difference between mean radial log10 power spectra. | blur/high-frequency positive controls; quarter-turn invariance |
| Feature distribution · MMD² | set | MMD² | Biased non-negative Gaussian-kernel MMD² in a training-fitted feature space. | training-EOF excursion |
| Feature density departure | set | absolute density departure | Absolute departure from Naeem-style feature-space density=1. | training-EOF excursion |
| Feature coverage deficit | set | fraction | Naeem-style real-manifold coverage deficit (1 - coverage). | feature-mode removal |
| Feature diversity departure | set | relative departure | Absolute relative departure in mean feature-space pairwise distance. | feature contraction and expansion |
