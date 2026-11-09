from .base_spatial_comp import BaseSpatial_PriorComp


class IsotropicSpatial_PriorComp(BaseSpatial_PriorComp):
    """Uniform sky map: log density 0, i.e. 1 per deg^2, so the spectral component carries the flux per deg^2."""

    @staticmethod
    def iso_logfunc(lonlatgrid, *args, **kwargs):
        return (lonlatgrid[..., 0] * 0)[None]     # (1, lon, lat)

    def __init__(self, *args, **kwargs):
        super().__init__(logfunc=self.iso_logfunc, *args, **kwargs)
