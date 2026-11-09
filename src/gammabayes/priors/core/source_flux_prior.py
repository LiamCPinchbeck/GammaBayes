from .discrete_logprior import DiscreteLogPrior
from scipy.special import logsumexp
import numpy as np
from gammabayes.samplers import  integral_inverse_transform_sampler
from gammabayes.utils import iterate_logspace_integration, construct_log_dx_mesh
from gammabayes import update_with_defaults, GammaObs, GammaBinning, GammaLogExposure

from astropy import units as u
import matplotlib.pyplot as plt
import warnings, logging
import h5py, pickle
from icecream import ic

class SourceFluxDiscreteLogPrior(DiscreteLogPrior):
    """A differential flux multiplied by the exposure.

    ``log_flux_function(grid, **params)`` returns the log differential flux
    [TeV^-1 cm^-2 s^-1 deg^-2] on the geometry's grid; the prior adds the log exposure from
    ``log_exposure_map`` (a :class:`~gammabayes.core.exposure.GammaLogExposure`, or one built from
    ``irf_loglike``, ``pointing_dirs`` and ``live_times``) to obtain log expected counts.  The exposure
    on the geometry's own grid is evaluated once and cached per exposure object.
    """

    def __init__(self,
                 log_flux_function: callable=None, 
                 binning_geometry: GammaBinning = None,
                 irf_loglike=None,
                 log_exposure_map=None, 
                 pointing_dirs=None, 
                 live_times=None,
                 log_scaling_factor=0.,
                 *args,
                 **kwargs
                 ):
        
        
        self.log_flux_function = log_flux_function

        self.irf_loglike = irf_loglike
        
        if binning_geometry is None:
            if hasattr(irf_loglike, "binning_geometry"):
                self.binning_geometry = self.irf_loglike.binning_geometry
            elif hasattr(log_exposure_map, "binning_geometry"):
                self.binning_geometry = log_exposure_map.binning_geometry
            else:
                raise ValueError("Binning geometry not given and cannot be extracted from other inputs")
        else:
            self.binning_geometry = binning_geometry

        if log_exposure_map is None:
            self.log_exposure_map = GammaLogExposure(binning_geometry=self.binning_geometry, 
                                                    irfs=self.irf_loglike,
                                                    log_exposure_map=log_exposure_map, 
                                                    pointing_dirs=pointing_dirs, 
                                                    live_times=live_times,)
        else:
            self.log_exposure_map = log_exposure_map

        super().__init__(
                 rate_tensor_logfunc=self.log_rate_source_function, 
                 binning_geometry=self.binning_geometry,
                 *args, 
                 **kwargs)

        if pointing_dirs is not None:

            self.pointing_dirs = pointing_dirs
        else:
            self.pointing_dirs = self.log_exposure_map.pointing_dirs

        if live_times is not None:
            self.live_times = live_times
        else:
            self.live_times = self.log_exposure_map.live_times

        

    def _log_exposure_on(self, grid, log_exposure_map=None):
        """Exposure evaluated on ``grid`` (cached for the geometry's own grid).
        The cache is keyed on the exposure *object*; if you mutate an exposure in
        place (``add_single_exposure``) call ``self._cached_log_exposure = None``."""
        if log_exposure_map is None:
            log_exposure_map = self.log_exposure_map
        own = (grid is getattr(self.binning_geometry, "_shaped_grid", None))
        if own and getattr(self, "_cached_log_exposure", None) is not None \
                and self._cached_log_exposure[0] is log_exposure_map:
            return self._cached_log_exposure[1]
        le = log_exposure_map(grid[..., 0].flatten(), grid[..., 1].flatten(), grid[..., 2].flatten()).reshape(grid.shape[:-1])
        if own:
            self._cached_log_exposure = (log_exposure_map, le)
        return le

    def log_rate_source_function(self, grid, log_exposure_map:GammaLogExposure=None, *args, **kwargs):
        """log flux (B, E, lon, lat) + log exposure (E, lon, lat) -> (B, E, lon, lat)."""
        return self.log_flux_function(grid, *args, **kwargs) + self._log_exposure_on(grid, log_exposure_map)