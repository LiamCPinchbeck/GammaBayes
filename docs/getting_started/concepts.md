# Concepts

This page summarises the conventions of the 2.x API.  Each item links to the tutorial that shows it
in code.

## Everything is a log-rate cube on a `GammaBinning`

A {class}`~gammabayes.core.binning_geometry.GammaBinning` is a regular 3-D grid
`(energy, lon, lat)`: log-uniform in energy (TeV), uniform in Galactic longitude and latitude
(deg).  Every model object evaluates to **natural-log expected counts per bin** on that grid:

```python
prior.eval_log_on_geom(spectral_params={...}, spatial_params={...})   # (E, lon, lat)  or  (B, E, lon, lat)
```

Working in log space keeps 40 orders of magnitude of dynamic range in float32 and makes
mixtures a `logaddexp`.  *-inf* is a legitimate value (zero expected counts, e.g. outside the
effective-area table) and is handled by the operators.  → {doc}`../tutorials/01_BinningGeometry`

## Flux × exposure = rate

A source is described by its differential flux
$\phi(E, l, b)$ in TeV⁻¹ cm⁻² s⁻¹ deg⁻².  {class}`~gammabayes.core.exposure.GammaLogExposure`
holds $\log[A_\mathrm{eff}(E,\theta)\, t_\mathrm{live}\, \Delta E\, \Delta l\, \Delta b]$ on the same
grid, so a {class}`~gammabayes.priors.SourceFluxDiscreteLogPrior` evaluates

$$\log \mu_{ijk} = \log \phi(E_i, l_j, b_k) + \log X_{ijk}.$$

Separable sources use {class}`~gammabayes.priors.TwoCompFluxPrior`:
$\log\phi = \log S(E) + \log M(l,b)$ with a **spectral** component (`PowerLaw`,
`LogParabola`, DM spectra, ...) and a **spatial** component (`PointSpatial_PriorComp`,
`GaussianSpatial_PriorComp`, `IsotropicSpatial_PriorComp`, DM `DM_Profile` J-factors, ...).
→ {doc}`../tutorials/02_IRFs_and_Exposure`, {doc}`../tutorials/03_Source_Components`

## How batching works

Every parameter of every component may be a Python scalar, a 0-d tensor, **or a 1-d tensor of
shape `(B,)`**.  Parameters passed together are broadcast to a common `B` (scalars count as
`B = 1`), and the component returns a tensor with a leading batch axis:

| component | input | output |
|---|---|---|
| spectral | `energy (E,)` | `(B, E)` |
| spatial | `lonlat (lon, lat, 2)` | `(B, lon, lat)` |
| flux prior | geometry grid | `(B, E, lon, lat)` |

`DiscreteLogPrior.eval_log_on_geom(squeeze=True)` (the default) drops the batch axis when
`B == 1`, so pre-2.x call sites keep working; pass `squeeze=False` to always get it.
Helpers live in {mod}`gammabayes.core.batching` (`as_param`, `batch_size`, `promote_batch`,
`iter_batch_chunks`).  → {doc}`../tutorials/03_Source_Components`,
{doc}`../tutorials/06_Likelihood_Batching_Autograd`

Everything is should be written with plain torch functions/methods/classes, so autograd ***should*** 
works through the whole framework, including the J-factor line-of-sight integral (gradient-checkpointed 
per chunk) and the IRF operator.

## True space vs reconstructed space

The instrument smears true energies and directions.  GammaBayes 2.1 models this with a
**forward operator** {class}`~gammabayes.utils.irf_forward.IRFForwardConv`:

$$\lambda_r = \sum_t P(r\,|\,t)\,\mu_t + \mathrm{CCR}_r,$$

where $\mu_t$ are expected counts on a **padded true grid** (`GammaBinning.padded(pads)`, a copy
of the analysis grid extended by the kernel half-width on every side) and $P(r|t)$ is a
bin-integrated Gaussian in $\log_{10}E$ (energy resolution in dex) and in $l$, $b$ (PSF
$\sigma$ in deg, optionally a function of true energy).  Each kernel row sums to one, so
**counts are conserved** apart from physical migration out of the window; the "valid"
convolution returns exactly the analysis grid.  The charged-cosmic-ray background (CCR) is already
a reconstructed-space quantity and is added *after* the operator.


## Units and conventions

* energy in TeV, angles in deg, $A_\mathrm{eff}$ in cm², live time in **seconds** internally
  (astropy quantities are converted), fluxes per deg².  DM densities are in TeV cm⁻³, so the
  differential J-factor comes out in TeV² cm⁻⁵ deg⁻² (multiply by 10⁶ for GeV² cm⁻⁵).
* DM annihilation flux prefactor $\langle\sigma v\rangle / (8\pi\, m_\chi^2 \cdot s)$ with the
  `symmetry_factor` $s = 1$ for self-conjugate (Majorana) particles; use `s = 2` for Dirac.
* The PSF is a 2-D Gaussian: $\sigma = R_{68} / \sqrt{-2\ln(1-0.68)} = R_{68}/1.51$.
* Energy resolution is $\sigma(\log_{10} E_\mathrm{reco} - \log_{10} E_\mathrm{true})$ in dex
  (CTAO prod5 South ≈ 0.09 dex above ~0.3 TeV).
