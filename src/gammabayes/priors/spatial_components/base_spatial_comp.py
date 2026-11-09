from gammabayes.core.batching import as_param, promote_batch
from functools import partial
import torch


class BaseSpatial_PriorComp:
    """Wraps a log-spatial function ``logfunc(lonlatgrid, **params)``.

    Batching: ``lonlatgrid`` is ``(lon, lat, 2)``; each parameter is a
    scalar or ``(B,)`` tensor; the return value is ``(B, lon, lat)``
    (``B = 1`` for all-scalar / parameter-free maps).  Subclasses build
    parameters with ``as_param(p, 2)`` so they broadcast as ``(B, 1, 1)``.
    """

    def __init__(self, logfunc, *args, **kwargs):
        self.__input_logfunc = logfunc

    def __call__(self, lonlatgrid, *args, **kwargs):
        out = self.__input_logfunc(lonlatgrid, *args, **kwargs)
        return promote_batch(out, 2)


class PointSpatial_PriorComp(BaseSpatial_PriorComp):
    """Point source at (``pos_lon``, ``pos_lat``) [deg]: the unit flux is deposited bilinearly on the four
    pixel centres around the position (log density per deg^2), which keeps the map differentiable in
    the position.  Positions may be scalars or ``(B,)``."""

    @staticmethod
    def log_point_mask(lonlatgrid, pos_lon, pos_lat, *args, **kwargs):
        L = lonlatgrid[..., 0]
        B_ = lonlatgrid[..., 1]
        # pixel steps from the sorted unique axis values (layout-agnostic)
        lu, bu = torch.unique(L), torch.unique(B_)
        dlon = (lu[1] - lu[0]).abs()
        dlat = (bu[1] - bu[0]).abs()
        pos_lon = as_param(pos_lon, 2, like=L)
        pos_lat = as_param(pos_lat, 2, like=L)
        # bilinear (tent) deposition onto the neighbouring pixel centres;
        # sums to 1 over the grid, differentiable in pos_lon / pos_lat
        w = ((1. - (L[None] - pos_lon).abs() / dlon).clamp(min=0.)
             * (1. - (B_[None] - pos_lat).abs() / dlat).clamp(min=0.))
        return torch.log(w.clamp(min=1e-32) / (dlon * dlat))   # floor: finite grad, e^-87 ~ 0 physically

    def __init__(self, pos_lon=0, pos_lat=0, *args, **kwargs):
        self.__log_point_mask = partial(self.log_point_mask, pos_lon=pos_lon, pos_lat=pos_lat)
        super().__init__(logfunc=self.__log_point_mask, *args, **kwargs)
