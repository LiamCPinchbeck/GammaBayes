# Migrating from 0.x

GammaBayes 2.1 is a port of the model layer to PyTorch with batching.  This page lists what
changed for code and notebooks written against 0.x.

## Maintained vs legacy

| area | 2.x status | modules |
|---|---|---|
| geometry, exposure, interpolation, batching | **torch, maintained** | `core.binning_geometry`, `core.exposure`, `core.reg_interpolator`, `core.batching` |
| source components and priors | **torch, maintained** | `priors.core`, `priors.spectral_components`, `priors.spatial_components`, `priors.astro_sources.isotropic_emission`, `priors.astro_sources.diffuse_emission.galprop_outputs` |
| dark matter | **torch, maintained** | `dark_matter.density_profiles`, `dark_matter.PPPC_Tables` (PPPC, CosmiXs, QCDUNC readers), `dark_matter.single_channel_prior`, `dark_matter.custom_ratios_model` |
| IRFs and operators | **torch, maintained** | `response_functions.core` (`IRFExtractor`, kernels), `utils.irf_forward`, `utils.log_space_conv_class` |
| event-list data classes | legacy (numpy + astropy units) | `core.gamma_cube` (`GammaObs`, `GammaObsCube`) |
| parameter bookkeeping | legacy | `core.parameter_class`, `core.parameter_set_class`, `core.parameter_set_collection_class` |
| nuisance scans / mixtures / reweighting | legacy | `hyper_inference.*`, `samplers.*` |
| catalogue sources, Fermi-Gaggero diffuse | legacy | `priors.astro_sources.localised_sources`, `priors.astro_sources.diffuse_emission.fermi_gaggero_diffuse` |
| CLI / SLURM helpers | legacy | `utils.cli.*`, `utils.config_utils`, `utils.event_axes` |

"Legacy" modules import, but they expect astropy-`Quantity` axes on the geometry and numpy
arrays from the priors, neither of which the 2.x core produces.  They are documented in the
API reference for completeness only.  The classes the 0.x tutorials revolved around
(`IRF_LogLikelihood`, `ObsFluxDiscreteLogPrior`, `High_Level_Analysis`, `SimulationContainer`,
`FermiGaggeroDiffusePrior`, `DM_ContinuousEmission_Spectrum`, `CombineDMComps`) **no longer
exist**.

## API changes

* `GammaBinning` axes are plain `torch.Tensor`s (TeV, deg), no units attached.
  `GammaBinning.from_edges(...)` builds a geometry from the edges; `geom.padded(pads)` gives
  the true-space grid for the forward operator; geometries compare equal when their edges do.
* Component parameters are keyword arguments at **call** time, scalar or `(B,)`:
  `comp(energy, index=torch.tensor([2.0, 2.5]))`.  Constructor values are defaults.
* `SingleDMSpectralComp` requires `mass=` at call time (no implicit 1 TeV).
* `CustomDMRatiosModel.eval_log_on_geom(spectral_params={"mass", "sigmav", "ratios", ["kappa"], ["delta"]}, spatial_params={"alpha"|"gamma", ["r_s"]})`;
  `ratios` is a dict or a `(B, n_ch)` tensor.
* `DiscreteLogPrior.eval_log_on_geom(squeeze=True)` returns `(E, lon, lat)` for `B == 1` and
  `(B, E, lon, lat)` otherwise.
* `GammaLogExposure(binning_geometry, irfs, pointing_dirs=[[l, b]], live_times=<Quantity | s>)`;
  live times are stored in seconds and the map includes the bin volume
  (`apply_bin_widths=True`).
* IRF smearing: build `IRFForwardConv(energy_kernel, spatial_kernel)` with
  `make_forward_irf_kernels(...)`, evaluate the model on `geom.padded(op.pads)` and call the
  operator; add the CCR afterwards.  `LogSumExpConv3D` keeps the old behaviour.
* `IRFExtractor(..., irf_time_seconds=18000)`: new explicit choice of the prod5 IRF
  optimisation (0.5 h / 5 h / 50 h).  `observation_time` does not select the file, and never did;
  the default reproduces the historical 5 h choice.

## Numerical differences to expect

* **Dependency stack**: moving from the pinned 0.x stack (astropy 5, gammapy 1.2, numpy 1) to
  the current one changes absolute rates at the per-cent level (interpolation and unit handling
  in several libraries); this is well below the statistical and systematic uncertainties of a GC
  analysis but will show up in bit-level comparisons with 0.x outputs.
* **float32**: the model floor is ~0.05 nats absolute plus ~1e-6·|log p|; set
  `torch.set_default_dtype(torch.float64)` for regression tests.
