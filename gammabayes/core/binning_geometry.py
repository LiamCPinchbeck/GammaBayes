import torch
import pickle

class GammaBinning:
    """Base lass to handle different coordinate dimensionalities and shapes. 
    e.g. 3D and 6D cartesian and ( 1D ) x ( 2D hexagonal ) geometries."""

    def __init__(self, 
            energy_res, energy_min, energy_max, 
            lon_res, lon_min, lon_max, 
            lat_res, lat_min, lat_max, 
            *args, **kwargs):
        energy_res = torch.tensor(energy_res)
        energy_min = torch.tensor(energy_min)
        energy_max = torch.tensor(energy_max)

        lon_res = torch.tensor(lon_res)
        lon_min = torch.tensor(lon_min)
        lon_max = torch.tensor(lon_max)

        lat_res = torch.tensor(lat_res)
        lat_min = torch.tensor(lat_min)
        lat_max = torch.tensor(lat_max)


        log10_E_min = torch.log10(energy_min)
        log10_E_max = torch.log10(energy_max)
        self.num_energy_bins = int((log10_E_max-log10_E_min)*energy_res+1)

        self.num_lon_bins = int((lon_max-lon_min)*lon_res+1)
        self.num_lat_bins = int((lat_max-lat_min)*lat_res+1)


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
    def axes(self):
        return [self.energy_axis, self.lon_axis, self.lat_axis]

    @property
    def axes_mesh(self):
        return torch.meshgrid(self.energy_axis, self.lon_axis, self.lat_axis, indexing='ij')

    @property
    def axes_dim(self):
        return (*(len(axis) for axis in self.axes), )

    @property
    def spatial_axes(self):
        return [self.lon_axis, self.lat_axis]
    
