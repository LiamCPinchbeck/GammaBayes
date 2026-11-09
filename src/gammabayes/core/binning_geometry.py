import torch
import pickle

class GammaBinning:
    """Regular (energy, longitude, latitude) analysis grid.

    The energy axis is log-uniform between ``energy_min`` and ``energy_max`` [TeV] with ``energy_res``
    bins per decade; the sky axes are uniform in Galactic longitude / latitude [deg] with ``lon_res`` /
    ``lat_res`` bins per degree.  ``energy_axis``, ``lon_axis`` and ``lat_axis`` hold the bin centres
    (the geometric centre in energy), ``*_edges`` the edges, ``bin_width_mat`` the bin volumes
    dE x dlon x dlat and ``shaped_grid`` the ``(E, lon, lat, 3)`` array of centre coordinates that model
    components receive.  Tensors are created on the default torch device and dtype.

    ``from_edges`` builds a geometry from explicit edges; ``padded`` returns the enlarged true-space grid
    used by :class:`gammabayes.utils.irf_forward.IRFForwardConv`.  Two geometries compare equal when
    their edges agree.
    """

    def __init__(self, 
            energy_res, energy_min, energy_max, 
            lon_res, lon_min, lon_max, 
            lat_res, lat_min, lat_max, 
            *args, **kwargs):
        energy_res = torch.tensor(energy_res)
        energy_min = torch.tensor(energy_min)
        energy_max = torch.tensor(energy_max)

        self.lon_res = torch.tensor(lon_res)
        lon_min = torch.tensor(lon_min)
        lon_max = torch.tensor(lon_max)

        self.lat_res = torch.tensor(lat_res)
        lat_min = torch.tensor(lat_min)
        lat_max = torch.tensor(lat_max)


        log10_E_min = torch.log10(energy_min)
        log10_E_max = torch.log10(energy_max)
        self.num_energy_bins = int((log10_E_max-log10_E_min)*energy_res+1)

        self.num_lon_bins = int((lon_max-lon_min)*self.lon_res+1)
        self.num_lat_bins = int((lat_max-lat_min)*self.lat_res+1)


        self.energy_edges = 10**torch.linspace(log10_E_min, log10_E_max, self.num_energy_bins)
        self.energy_axis = 10**(
            0.5*(
                torch.log10(self.energy_edges[1:])+torch.log10(self.energy_edges[:-1])
                )
            )

        self.log_energy_edges = torch.log10(self.energy_edges)
        self.lon_edges = torch.linspace(lon_min, lon_max, self.num_lon_bins)

        self.lon_axis = 0.5*(self.lon_edges[1:]+self.lon_edges[:-1])

        self.lat_edges = torch.linspace(lat_min, lat_max, self.num_lat_bins)

        self.lat_axis = 0.5*(self.lat_edges[1:]+self.lat_edges[:-1])


        self.spatial_centre = torch.tensor([(lon_max+lon_min)/2, (lat_max+lat_min)/2])

        self.bin_width_mat = self._construct_bin_widths()

        self.axes = (self.energy_axis, self.lon_axis, self.lat_axis)
        self.axes_edges = (self.energy_edges, self.lon_edges, self.lat_edges)

    @classmethod
    def from_edges(cls, energy_edges, lon_edges, lat_edges, lon_res=None, lat_res=None):
        """Build directly from bin edges (no ``int(range*res + 1)`` truncation), e.g. a padded
        copy of another geometry.  Centres follow the constructor's conventions."""
        self = cls.__new__(cls)
        self.energy_edges = torch.as_tensor(energy_edges)
        self.lon_edges = torch.as_tensor(lon_edges)
        self.lat_edges = torch.as_tensor(lat_edges)
        self.num_energy_bins = len(self.energy_edges)
        self.num_lon_bins = len(self.lon_edges)
        self.num_lat_bins = len(self.lat_edges)
        self.lon_res = torch.tensor(lon_res if lon_res is not None else 1 / float(self.lon_edges[1] - self.lon_edges[0]))
        self.lat_res = torch.tensor(lat_res if lat_res is not None else 1 / float(self.lat_edges[1] - self.lat_edges[0]))
        self.log_energy_edges = torch.log10(self.energy_edges)
        self.energy_axis = 10 ** (0.5 * (self.log_energy_edges[1:] + self.log_energy_edges[:-1]))
        self.lon_axis = 0.5 * (self.lon_edges[1:] + self.lon_edges[:-1])
        self.lat_axis = 0.5 * (self.lat_edges[1:] + self.lat_edges[:-1])
        self.spatial_centre = torch.stack([(self.lon_edges[0] + self.lon_edges[-1]) / 2,
                                           (self.lat_edges[0] + self.lat_edges[-1]) / 2])
        self.bin_width_mat = self._construct_bin_widths()
        self.axes = (self.energy_axis, self.lon_axis, self.lat_axis)
        self.axes_edges = (self.energy_edges, self.lon_edges, self.lat_edges)
        return self





        
    # -- equality on the edges (exposures / priors built on separately constructed but identical
    #    geometries used to compare unequal because the default __eq__ is identity)
    def __eq__(self, other):
        if not isinstance(other, GammaBinning):
            return NotImplemented
        return all(a.shape == b.shape and torch.allclose(a.to(b.dtype), b, rtol=1e-6, atol=1e-9)
                   for a, b in zip(self.axes_edges, other.axes_edges))

    __hash__ = object.__hash__

    def __repr__(self):
        e, l, b = self.axes_edges
        return (f"GammaBinning(E: {len(e)-1} bins {float(e[0]):.3g}-{float(e[-1]):.3g} TeV, "
                f"lon: {len(l)-1} bins {float(l[0]):.3g}..{float(l[-1]):.3g} deg, "
                f"lat: {len(b)-1} bins {float(b[0]):.3g}..{float(b[-1]):.3g} deg)")

    def padded(self, pads):
        """Geometry extended by ``pads = (nE, nLon, nLat)`` whole bins on every side (same bin
        widths; log-uniform in energy).  The interior edges are this geometry's edges exactly, so a
        'valid' convolution over the padded cube returns ``self``.  This is the true-space grid for
        :class:`gammabayes.utils.IRFForwardConv`."""
        pE, pL, pB = (int(p) for p in pads)

        def ext(e, p):
            if p == 0:
                return e
            d = e[1] - e[0]
            k = torch.arange(1, p + 1, dtype=e.dtype, device=e.device)
            return torch.cat([e[0] - d * k.flip(0), e, e[-1] + d * k])

        e = self.energy_edges
        lE = ext(torch.log10(e.detach().cpu().double()), pE)            # float64 on the CPU (MPS has no float64)
        return GammaBinning.from_edges((10 ** lE).to(dtype=e.dtype, device=e.device),
                                       ext(self.lon_edges, pL), ext(self.lat_edges, pB),
                                       lon_res=float(self.lon_res), lat_res=float(self.lat_res))

    def _construct_bin_widths(self):
        self.energy_bin_widths = self.energy_edges[1:] - self.energy_edges[:-1]
        self.lon_bin_widths = self.lon_edges[1:]-self.lon_edges[:-1]
        self.lat_bin_widths = self.lat_edges[1:]-self.lat_edges[:-1]


        bin_matrix = self.energy_bin_widths[:, None, None]*self.lon_bin_widths[None, :, None]*self.lat_bin_widths[None, None, :]

        return bin_matrix




    @property
    def centre_shape(self):
        return torch.Size(
            (
                self.energy_axis.shape[0],
                self.lon_axis.shape[0],
                self.lat_axis.shape[0],
                ))


    @property
    def grid(self):
        if not hasattr(self, "_grid"):

            meshes = torch.meshgrid(
                (self.energy_axis, self.lon_axis, self.lat_axis),
                indexing='ij')

            self._grid = torch.stack([mesh.flatten() for mesh in meshes], axis=1)

        return self._grid


    @property
    def shaped_grid(self):
        if not hasattr(self, "_shaped_grid"):
            self._shaped_grid = self.grid.reshape((*self.centre_shape, 3))
        
        return self._shaped_grid


    def grid_idx_to_vals(self, idx):
        return self.grid[idx]


    def _build_lookup_dict(self):
        """Creates a mapping from grid rows to indices (as tuple keys)."""
        grid = self.grid
        # Round to avoid floating point fuzziness
        rounded_grid = torch.round(grid * 1e6).to(torch.int64)
        keys = [tuple(row.tolist()) for row in rounded_grid]
        self._lookup_dict = {k: i for i, k in enumerate(keys)}


    def values_to_grid(self, values):
        if not hasattr(self, "_lookup_dict"):
            self._build_lookup_dict()

        if values.ndim == 1:
            values = values[None, :]

        values = torch.round(values * 1e6).to(torch.int64)

        indices = []
        for row in values:
            key = tuple(row.tolist())
            if key in self._lookup_dict:
                indices.append(self._lookup_dict[key])
            else:
                raise ValueError(f"Value {row} not found in grid.")

        return torch.tensor(indices, dtype=torch.long, device=values.device)




    @property
    def axes_mesh(self):
        return torch.stack((*torch.meshgrid(self.energy_axis, self.lon_axis, self.lat_axis, indexing='ij'),), dim=0)

    @property
    def axes_dim(self):
        return (*(len(axis) for axis in self.axes), )

    @property
    def spatial_axes(self):
        return [self.lon_axis, self.lat_axis]


    @property
    def shape(self):
        return self.axes_dim


    


    
