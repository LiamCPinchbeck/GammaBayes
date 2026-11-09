# Tutorials

Recommended order:

1. **01_BinningGeometry** - the analysis grid, padded true-space grids, geometry helpers.
2. **02_IRFs_and_Exposure** - loading CTAO IRFs, effective area, energy-dependent PSF width,
   CCR background, building the exposure cube.
3. **03_Source_Components** - spectral and spatial components, `TwoCompFluxPrior`, how batching
   works, static templates.
4. **04_DarkMatter** - density profiles and J-factors, annihilation spectra (PPPC / CosmiXs,
   `kappa` and `delta`), `CustomDMRatiosModel`, expected DM counts.
5. **05_IRF_Forward_Operator** - bin-integrated kernels, count conservation, energy-dependent PSF,
   comparison with the legacy `LogSumExpConv3D`.
6. **06_Likelihood_Batching_Autograd** - an end-to-end toy search: forward model, Asimov and
   Poisson data, batched Poisson likelihood with autograd, nuisance-parameter profiling over the
   $(m_\chi, \langle\sigma v\rangle)$ grid with an analytic template-fit Newton method (with and
   without a 1 % per-energy-bin background systematic), median expected 95 % CL upper limits versus
   mass, detection significance and confidence region for an injected thermal-relic signal, Laplace
   approximation versus the profile, one Poisson realisation.

All notebooks run on a laptop CPU in a few minutes (set `GAMMABAYES_DOCS_FAST=1` for smaller
grids; `GAMMABAYES_DOCS_DEVICE=cpu|cuda|mps` selects the device) and use only the data shipped with
the package.  Notebook 06 runs on the CPU in float64 even when MPS is available.

## Run times

<!-- timings:start -->
Last executed on **cpu / torch.float64** (2026-11-09); every code cell shows its own wall time at the end of its output.

| notebook | total wall time |
|---|---|
| 01_BinningGeometry.ipynb | 3.0 s |
| 02_IRFs_and_Exposure.ipynb | 4.8 s |
| 03_Source_Components.ipynb | 3.8 s |
| 04_DarkMatter.ipynb | 7.5 s |
| 05_IRF_Forward_Operator.ipynb | 5.3 s |
| 06_Likelihood_Batching_Autograd.ipynb | 12.5 s |
<!-- timings:end -->
