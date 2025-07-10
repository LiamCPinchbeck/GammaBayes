import numpy as np
from os import path
from gammabayes.priors.core.discrete_logprior import DiscreteLogPrior
from gammabayes.priors.core.source_flux_prior import SourceFluxDiscreteLogPrior
from gammabayes.core import GammaLogExposure, GammaBinning
import time

from icecream import ic

class TwoCompFluxPrior(SourceFluxDiscreteLogPrior):
    def __init__(self, 
                 spectral_comp, 
                 spatial_comp, 
                 *args, **kwargs
                 ):


        self.spectral_comp    = spectral_comp
        self.spatial_comp     = spatial_comp

        super().__init__(
            log_flux_function = self.log_flux_function, 
            *args, **kwargs
            )
            

    def log_flux_function(self, grid, spectral_params={}, spatial_params={}):

        # Extracting the unique energy values
        energy_vals  = grid[...,0,0,0] 

        # Extracting the unique spatial values
        spatial_vals = grid[0, ..., 1:]

        energy_comp_vals    = self.spectral_comp(energy_vals, **spectral_params)
        spatial_comp_vals   = self.spatial_comp(spatial_vals, **spatial_params)

        output_mat = energy_comp_vals[:, None, None] + spatial_comp_vals[None, :, :]

        return output_mat
        

