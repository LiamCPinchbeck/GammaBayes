import torch
from .base_spatial_comp import BaseSpatial_PriorComp
from gammabayes.core.batching import as_param
from functools import partial


class GaussianSpatial_PriorComp(BaseSpatial_PriorComp):
    """Bivariate Gaussian sky map (log density per deg^2) with centre (``pos_lon``, ``pos_lat``), widths
    ``sigma_lon`` / ``sigma_lat`` [deg] and correlation ``rho``; all parameters scalar or ``(B,)``."""

    @staticmethod
    def gaussian_logfunc(lonlatgrid, pos_lon, pos_lat, sigma_lon, sigma_lat, rho, normalisation=1., *args, **kwargs):
        L = lonlatgrid[..., 0][None]
        B_ = lonlatgrid[..., 1][None]
        pos_lon, pos_lat, sigma_lon, sigma_lat, rho, normalisation = (
            as_param(v, 2, like=L) for v in (pos_lon, pos_lat, sigma_lon, sigma_lat, rho, normalisation))

        longitude_factor = (L - pos_lon) / sigma_lon
        latitude_factor = (B_ - pos_lat) / sigma_lat

        prefactor = 1 / (2 * torch.pi * sigma_lon * sigma_lat * torch.sqrt(1 - rho ** 2))
        exponent_prefactor = -1 / (2 * (1 - rho ** 2))
        exponent_body = longitude_factor ** 2 - 2 * rho * longitude_factor * latitude_factor + latitude_factor ** 2

        output = torch.log(normalisation * prefactor) + exponent_prefactor * exponent_body
        return torch.where(torch.isnan(output), -torch.inf, output)

    def __init__(self, pos_lon=0, pos_lat=0,
                 sigma_lon=0.1, sigma_lat=0.1, rho=0,
                 *args, **kwargs):
        self.logfunc = partial(self.gaussian_logfunc,
                               pos_lon=pos_lon, pos_lat=pos_lat,
                               sigma_lon=sigma_lon, sigma_lat=sigma_lat,
                               rho=rho)
        super().__init__(logfunc=self.logfunc, *args, **kwargs)
