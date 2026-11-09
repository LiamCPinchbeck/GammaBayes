API reference
=============

The reference is for the **maintained torch core** (everything a 2.x analysis touches).

.. toctree::
   :maxdepth: 2

   core
   priors
   dark_matter
   response_functions
   operators
   utils

Quick index
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Task
     - Classes and functions
   * - define the analysis grid
     - :class:`gammabayes.core.binning_geometry.GammaBinning` (``from_edges``, ``padded``)
   * - load CTAO IRFs, get :math:`A_\mathrm{eff}`, PSF width vs energy, CCR background
     - :class:`gammabayes.response_functions.core.irf_extractor_class.IRFExtractor`
   * - build the exposure cube :math:`A_\mathrm{eff}\,t\,\Delta V`
     - :class:`gammabayes.core.exposure.GammaLogExposure`
   * - describe an astrophysical source
     - :class:`gammabayes.priors.core.two_comp_prior.TwoCompFluxPrior` with a spectral and a spatial component
   * - use a fixed template (GALPROP, HESS maps, CCR)
     - :class:`gammabayes.priors.core.static_rate_template.StaticSourceDistTemplate`, :func:`gammabayes.priors.astro_sources.diffuse_emission.galprop_outputs.get_galprop_static_source_flux_prior`
   * - model dark-matter annihilation
     - :class:`gammabayes.dark_matter.custom_ratios_model.CustomDMRatiosModel`, :class:`gammabayes.dark_matter.single_channel_prior.SingleChannelDMPrior`
   * - J-factors
     - :class:`gammabayes.dark_matter.density_profiles.DM_Profiles.Einasto_Profile`, :class:`gammabayes.dark_matter.density_profiles.DM_Profiles.GNFW_Profile`
   * - apply the instrument response
     - :class:`gammabayes.utils.irf_forward.IRFForwardConv` + :func:`gammabayes.utils.irf_forward.make_forward_irf_kernels`
   * - batch parameters / chunk work
     - :mod:`gammabayes.core.batching`
