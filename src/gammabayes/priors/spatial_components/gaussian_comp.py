import torch
from .base_spatial_comp import BaseSpatial_PriorComp
from functools import partial

class GaussianSpatial_PriorComp(BaseSpatial_PriorComp):


    @staticmethod
    def gaussian_logfunc(lonlatgrid, pos_lon, pos_lat, sigma_lon, sigma_lat, rho, normalisation=1., *args, **kwargs):

        longitude_factor = (lonlatgrid[..., 0]-pos_lon)/sigma_lon
        latitude_factor = (lonlatgrid[..., 1]-pos_lat)/sigma_lat

        prefactor = 1/(2*torch.pi*sigma_lon*sigma_lat*torch.sqrt(1-rho**2))
        exponent_prefactor = -1/(2*(1-rho**2))
        exponent_body = longitude_factor**2-2*rho*longitude_factor*latitude_factor+latitude_factor**2

        output = torch.log(normalisation*prefactor) + exponent_prefactor*exponent_body

        output = torch.where(torch.isnan(output), -torch.inf, output)

        return output    

    def __init__(self,pos_lon=0, pos_lat=0, 
                sigma_lon=0.1, sigma_lat=0.1, rho=0, 
                *args, **kwargs):

        self.logfunc = partial(self.gaussian_logfunc,
                                pos_lon=pos_lon, pos_lat=pos_lat, 
                                sigma_lon=sigma_lon, sigma_lat=sigma_lat, 
                                rho=rho)


        super().__init__(logfunc=self.logfunc, *args, **kwargs)