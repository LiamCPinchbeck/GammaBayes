from gammabayes.priors.core.source_flux_prior import SourceFluxDiscreteLogPrior
from gammabayes.core.batching import promote_batch


class TwoCompFluxPrior(SourceFluxDiscreteLogPrior):
    """Separable flux prior: log F(E, lon, lat) = log S(E) + log M(lon, lat), times the exposure.

    ``spectral_comp(energy, **spectral_params)`` returns ``(B1, E)`` and
    ``spatial_comp(lonlat, **spatial_params)`` returns ``(B2, lon, lat)``; they are combined by
    broadcasting, so ``B1, B2 in {1, B}``, and the log exposure is added.  ``eval_log_on_geom`` returns
    ``(B, E, lon, lat)`` log expected counts (``(E, lon, lat)`` for ``B == 1``).  Parameters not passed at
    call time take the components' constructor defaults.
    """

    def __init__(self, spectral_comp, spatial_comp, *args, **kwargs):
        self.spectral_comp = spectral_comp
        self.spatial_comp = spatial_comp
        super().__init__(log_flux_function=self.log_flux_function, *args, **kwargs)

    def log_flux_function(self, grid, spectral_params=None, spatial_params=None):
        spectral_params = spectral_params or {}
        spatial_params = spatial_params or {}

        energy_vals = grid[..., 0, 0, 0]      # (E,)
        spatial_vals = grid[0, ..., 1:]       # (lon, lat, 2)

        energy_comp_vals = promote_batch(self.spectral_comp(energy_vals, **spectral_params), 1)     # (B1, E)
        spatial_comp_vals = promote_batch(self.spatial_comp(spatial_vals, **spatial_params), 2)    # (B2, lon, lat)

        return energy_comp_vals[:, :, None, None] + spatial_comp_vals[:, None, :, :]
