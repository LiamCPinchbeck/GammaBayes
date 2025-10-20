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


    def __init__(self,
                 log_flux_function: callable=None, 
                 binning_geometry: GammaBinning = None,
                 irf_loglike=None,
                 log_exposure_map=None, 
                 pointing_dirs:np.ndarray[u.Quantity]=None, 
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

        

    def log_rate_source_function(self, grid, log_exposure_map:GammaLogExposure=None, *args, **kwargs):
        if log_exposure_map is None:
            log_exposure_map = self.log_exposure_map


        return self.log_flux_function(grid, *args, **kwargs) + log_exposure_map(grid[..., 0].flatten(), grid[..., 1].flatten(), grid[..., 2].flatten(), ).reshape(grid.shape[:-1])