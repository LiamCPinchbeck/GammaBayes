from astropy import units as u
import numpy as np
from gammabayes.priors.core.wrappers import _wrap_if_missing_keyword
from gammabayes import update_with_defaults

from .base_spatial_comp import BaseSpatial_PriorComp
from icecream import ic

class IsotropicSpatial_PriorComp(BaseSpatial_PriorComp):

    @staticmethod
    def iso_logfunc(lonlatgrid, *args, **kwargs):
        return lonlatgrid[..., 0]*0
    
    
    def __init__(self, *args, **kwargs):

        super().__init__(logfunc=self.iso_logfunc, *args, **kwargs)