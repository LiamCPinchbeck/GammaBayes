from astropy import units as u
from gammabayes.priors.core.wrappers import _wrap_if_missing_keyword
from gammabayes import update_with_defaults
from functools import partial


class BaseSpatial_PriorComp:

    def __init__(self, logfunc, *args, **kwargs):

        self._input_logfunc = logfunc


    def __call__(self, *args, **kwargs):


        return self._input_logfunc(*args, **kwargs)




class PointSpatial_PriorComp(BaseSpatial_PriorComp):

    @staticmethod
    def log_point_mask(lonlatgrid, pos_lon, pos_lat, *args, **kwargs):

        lon_vals = lonlatgrid[:, 0, 0]
        lat_vals = lonlatgrid[0, :, 0]

        lon_arg = torch.abs(lon_vals-pos_lon).argmin()
        lat_arg = torch.abs(lat_vals-pos_lat).argmin()

        point_mask_out = torch.zeros_like(lonlatgrid[:, :, 0])

        point_mask_out[lon_arg, lat_arg] = 1.

        return point_mask_out.log()

    def __init__(self,pos_lon=0, pos_lat=0,
                *args, **kwargs):

        self.__log_point_mask = partial(self.log_point_mask,
                                pos_lon=pos_lon, pos_lat=pos_lat)


        super().__init__(logfunc=self.__log_point_mask, *args, **kwargs)