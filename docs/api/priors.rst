Priors (source models)
======================

A *prior* in GammaBayes is a model for the expected counts of one source class on the binning
geometry.  The class hierarchy is

* :class:`~gammabayes.priors.core.discrete_logprior.DiscreteLogPrior` -- anything that can
  produce a log-rate cube (``eval_log_on_geom``), draw Poisson or categorical samples from it;
* :class:`~gammabayes.priors.core.source_flux_prior.SourceFluxDiscreteLogPrior` -- a
  differential flux times a :class:`~gammabayes.core.exposure.GammaLogExposure`;
* :class:`~gammabayes.priors.core.two_comp_prior.TwoCompFluxPrior` -- separable
  flux ``log S(E) + log M(lon, lat)`` built from one spectral and one spatial component;
* :class:`~gammabayes.priors.core.static_rate_template.StaticSourceDistTemplate` /
  :class:`~gammabayes.priors.core.static_rate_template.StaticDistTemplate` -- fixed tensors.

Core classes
------------

.. automodule:: gammabayes.priors.core.discrete_logprior

.. automodule:: gammabayes.priors.core.source_flux_prior

.. automodule:: gammabayes.priors.core.two_comp_prior

.. automodule:: gammabayes.priors.core.static_rate_template

.. automodule:: gammabayes.priors.core.wrappers
   :members: _wrap_if_missing_keyword

Spectral components
-------------------

All components take the energy axis ``(E,)`` in TeV and keyword parameters that are scalars or
``(B,)`` tensors, and return ``(B, E)`` natural-log differential flux.

.. automodule:: gammabayes.priors.spectral_components.base_spectral_comp

.. automodule:: gammabayes.priors.spectral_components.power_law_subclasses

.. automodule:: gammabayes.priors.spectral_components.log_parabola_funcs

Spatial components
------------------

All components take a ``(lon, lat, 2)`` grid of pixel centres in degrees and return
``(B, lon, lat)`` natural-log sky maps (per deg\ :sup:`2`).

.. automodule:: gammabayes.priors.spatial_components.base_spatial_comp

.. automodule:: gammabayes.priors.spatial_components.gaussian_comp

.. automodule:: gammabayes.priors.spatial_components.isotropic_map

Astrophysical sources
---------------------

.. automodule:: gammabayes.priors.astro_sources.isotropic_emission.isotropic_model

.. automodule:: gammabayes.priors.astro_sources.diffuse_emission.galprop_outputs
