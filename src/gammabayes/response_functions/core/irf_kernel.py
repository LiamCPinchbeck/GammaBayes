import torch
from torch.distributions import Normal, Categorical


def make_normal_kernel_with_std_and_bins(linear_axis, std, confidence_clearance=5):

    bin_width = linear_axis[1]-linear_axis[0]

    num_bins = int(confidence_clearance*std/bin_width)


    annoying=False
    if num_bins%2 ==0:
        annoying=True
        num_bins += 1

    kernel_pseudo_axis = torch.arange(-num_bins*bin_width, (num_bins+1)*bin_width, bin_width)

    if 2* num_bins + 1 != len(kernel_pseudo_axis) and annoying:
        kernel_pseudo_axis = kernel_pseudo_axis[:-1]


    base_normal_dist = Normal(loc=torch.tensor(0.), scale=std)


    return base_normal_dist.log_prob(kernel_pseudo_axis).exp()


loge10 = torch.log(torch.tensor(10.))

def make_irf_kernel(geom, log10energy_std, lon_std, lat_std, confidence_clearance=5):

    log10_energy_kernel_axis = make_normal_kernel_with_std_and_bins(
        torch.log10(geom.energy_axis), 
        log10energy_std/torch.log10(torch.tensor(torch.e)), confidence_clearance=confidence_clearance)

    longitude_kernel_axis = make_normal_kernel_with_std_and_bins(
        geom.lon_axis, lon_std, confidence_clearance=confidence_clearance)

    latitude_kernel_axis = make_normal_kernel_with_std_and_bins(
        geom.lat_axis, lat_std, confidence_clearance=confidence_clearance)

    irf_kernel = log10_energy_kernel_axis[:, None, None]*longitude_kernel_axis[None, :, None]*latitude_kernel_axis[None, None, :]

    return irf_kernel




class IRFKernelDist:

    def __init__(self, measured_geom, irf_extractor, 
                    log_energy_sigma, lon_sigma, lat_sigma, kernel_sigmas=5):
        self.irfs = irf_extractor
        self.binning_geometry = measured_geom

        self.log10_energy_axis = torch.log10(self.binning_geometry.energy_axis)

        self.log10_energy_diff = self.log10_energy_axis[1]-self.log10_energy_axis[0]
        self.lon_diff = self.binning_geometry.lon_axis[1]-self.binning_geometry.lon_axis[0]
        self.lat_diff = self.binning_geometry.lat_axis[1]-self.binning_geometry.lat_axis[0]

        self.kernel_sigmas = kernel_sigmas
        self.log_energy_sigma = log_energy_sigma
        self.lon_sigma = lon_sigma
        self.lat_sigma = lat_sigma

        self.num_energy_kernel_bins = torch.tensor(
                                                torch.ceil(
                                                    self.log_energy_sigma*kernel_sigmas/self.log10_energy_diff
                                                    ), 
                                                dtype=int)

        self.num_lon_kernel_bins = torch.tensor(
                                                torch.ceil(
                                                    self.lon_sigma*kernel_sigmas/self.lon_diff
                                                    ), 
                                                dtype=int)

        self.num_lat_kernel_bins = torch.tensor(
                                                torch.ceil(
                                                    self.lat_sigma*kernel_sigmas/self.lat_diff
                                                    ), 
                                                dtype=int)

        self._construct_kernel_axes()
        self._construct_kernels()
        self._construct_kernel_dists()




    def _construct_kernel_axes(self):
        self.energy_disp_kernel_axis = torch.arange(
            -self.num_energy_kernel_bins*self.log10_energy_diff, 
            (self.num_energy_kernel_bins+1)*self.log10_energy_diff,
            self.log10_energy_diff
            )


        self.lon_psf_kernel_axis = torch.arange(
            -self.num_lon_kernel_bins*self.lon_diff, 
            (self.num_lon_kernel_bins+1)*self.lon_diff,
            self.lon_diff
            )

        self.lat_psf_kernel_axis = torch.arange(
            -self.num_lat_kernel_bins*self.lat_diff, 
            (self.num_lat_kernel_bins+1)*self.lat_diff,
            self.lat_diff
            )

        self.lonlat_psf_kernel_mesh = torch.stack(
                                        torch.meshgrid(
                                            self.lon_psf_kernel_axis,
                                            self.lat_psf_kernel_axis,
                                            indexing='ij'
                                        ), dim=0)


    def _construct_kernels(self):
        self.energy_disp_kernel = -torch.log(2*torch.pi*self.log_energy_sigma**2) - 0.5*self.energy_disp_kernel_axis**2/self.log_energy_sigma**2
        self.energy_disp_kernel -= torch.logsumexp(self.energy_disp_kernel, dim=0)
        
        self.energy_disp_kernel = self.energy_disp_kernel.exp()

        self.lonlat_psf_kernel = (
            -0.5*self.lonlat_psf_kernel_mesh[0]**2/self.lon_sigma**2
            -0.5*self.lonlat_psf_kernel_mesh[1]**2/self.lat_sigma**2
            -torch.log(2*torch.pi*self.lon_sigma*self.lat_sigma)
        )
        self.lonlat_psf_kernel -= torch.logsumexp(self.lonlat_psf_kernel, dim=(0,1))

        self.lonlat_psf_kernel = self.lonlat_psf_kernel.exp()

        self.edisp_norm  = self.energy_disp_kernel.sum()
        self.psf_norm    = self.lonlat_psf_kernel.sum()


    def _construct_kernel_dists(self):

        self.edisp_index_cat = Categorical(probs=self.energy_disp_kernel)
        self.psf_index_cat = Categorical(probs=self.lonlat_psf_kernel.flatten())
        

    def _sample_indices(self, num_samples):
        energy_indices = self.edisp_index_cat.sample((num_samples,)) - self.num_energy_kernel_bins


        lonlat_indices = self.psf_index_cat.sample((num_samples,))

        lon_indices, lat_indices = torch.unravel_index(lonlat_indices, self.lonlat_psf_kernel.shape)
        lon_indices, lat_indices = lon_indices-self.num_lon_kernel_bins, lat_indices-self.num_lat_kernel_bins
        return energy_indices, lon_indices, lat_indices


    def sample_kernel(self, true_value_grid):
        num_samples = int(true_value_grid.sum())
        energy_indices, lon_indices, lat_indices = self._sample_indices(num_samples)

        flattened_coord_grid = self.binning_geometry.shaped_grid.reshape((-1, 3))

        flattened_meas_value_grid = torch.zeros_like(true_value_grid)
        meas_grid_shape = flattened_meas_value_grid.shape
        event_counter= 0 
        separate_counter = 0
        for energy_idx, energy_val in enumerate(self.binning_geometry.energy_axis):
            for lon_idx, lon_val in enumerate(self.binning_geometry.lon_axis):
                for lat_idx, lat_val in enumerate(self.binning_geometry.lat_axis):
                    

                    num_val = int(true_value_grid[energy_idx, lon_idx, lat_idx])

                    if num_val>0:
                        valset_energy_kernel_indices = energy_indices[event_counter:event_counter+num_val]
                        valset_lon_kernel_indices = lon_indices[event_counter:event_counter+num_val]
                        valset_lat_kernel_indices = lat_indices[event_counter:event_counter+num_val]


                        valset_energy_indices   = valset_energy_kernel_indices + energy_idx
                        valset_lon_indices      = valset_lon_kernel_indices + lon_idx
                        valset_lat_indices      = valset_lat_kernel_indices + lat_idx


                        good_energy_indices = torch.where(valset_energy_indices<0, False, True)
                        good_lon_indices    = torch.where(valset_lon_indices<0, False, True)
                        good_lat_indices    = torch.where(valset_lat_indices<0, False, True)

                        good_energy_indices2 = torch.where(valset_energy_indices>=meas_grid_shape[0], False, True)
                        good_lon_indices2    = torch.where(valset_lon_indices>=meas_grid_shape[1], False, True)
                        good_lat_indices2    = torch.where(valset_lat_indices>=meas_grid_shape[2], False, True)

                        indices_cleaned = good_energy_indices*good_lon_indices*good_lat_indices
                        indices_cleaned2 = good_energy_indices2*good_lon_indices2*good_lat_indices2

                        indices_cleaned = indices_cleaned*indices_cleaned2

                        valset_energy_indices   = valset_energy_indices[indices_cleaned]
                        valset_lon_indices      = valset_lon_indices[indices_cleaned]
                        valset_lat_indices      = valset_lat_indices[indices_cleaned]

                        separate_counter+=len(valset_energy_indices)
                        
                        flattened_meas_value_grid[valset_energy_indices, valset_lon_indices, valset_lat_indices]+=1

                        event_counter += num_val

        return flattened_meas_value_grid














# ---------------------------------------------------------------------------
# Energy-calibrated PSF kernels (A3)
# ---------------------------------------------------------------------------
def make_spatial_kernel_per_energy(geom, sigma_deg, confidence_clearance=5, pixel_floor=False):
    """Per-true-energy 2-D Gaussian PSF kernels on a common tap grid.

    sigma_deg: (E,) tensor of Gaussian widths [deg] (``IRFExtractor.psf_sigma_vs_energy``).
    The tap grid is sized by the LARGEST sigma (``confidence_clearance`` sigmas,
    odd number of taps); narrow planes just have most of their weight near the
    centre.  Each plane is normalised to unit sum.  Returns (E, K_lon, K_lat).

    pixel_floor adds Delta^2/12 per axis in quadrature (use it here OR in
    ``psf_sigma_vs_energy``, not both).
    """
    sigma = torch.as_tensor(sigma_deg, dtype=geom.lon_axis.dtype).reshape(-1)
    if sigma.shape[0] != len(geom.energy_axis):
        raise ValueError(f"sigma_deg has {sigma.shape[0]} entries, geometry has {len(geom.energy_axis)} energy bins")
    dlon = geom.lon_axis[1] - geom.lon_axis[0]
    dlat = geom.lat_axis[1] - geom.lat_axis[0]
    s_lon = torch.sqrt(sigma ** 2 + dlon ** 2 / 12) if pixel_floor else sigma
    s_lat = torch.sqrt(sigma ** 2 + dlat ** 2 / 12) if pixel_floor else sigma

    n_lon = int(torch.ceil(confidence_clearance * s_lon.max() / dlon))
    n_lat = int(torch.ceil(confidence_clearance * s_lat.max() / dlat))
    ax_lon = torch.arange(-n_lon, n_lon + 1, dtype=sigma.dtype) * dlon           # (K_lon,)
    ax_lat = torch.arange(-n_lat, n_lat + 1, dtype=sigma.dtype) * dlat           # (K_lat,)

    k_lon = torch.exp(-0.5 * (ax_lon[None, :] / s_lon[:, None]) ** 2)            # (E, K_lon)
    k_lat = torch.exp(-0.5 * (ax_lat[None, :] / s_lat[:, None]) ** 2)            # (E, K_lat)
    k = k_lon[:, :, None] * k_lat[:, None, :]
    return k / k.sum((1, 2), keepdim=True)


def make_irf_kernel_per_energy(geom, log10energy_std, sigma_deg, confidence_clearance=5, pixel_floor=False):
    """Separable IRF kernel with an energy-dependent PSF.

    Returns ``(energy_kernel (K_d,), spatial_kernel (E, K_lon, K_lat))`` in
    linear space, ready for ``LogSumExpConv3D(energy_kernel=..., spatial_kernel=...)``.
    The energy kernel is built exactly as in ``make_irf_kernel`` so the
    dispersion treatment is unchanged.
    """
    energy_kernel = make_normal_kernel_with_std_and_bins(
        torch.log10(geom.energy_axis),
        log10energy_std / torch.log10(torch.tensor(torch.e)), confidence_clearance=confidence_clearance)
    spatial_kernel = make_spatial_kernel_per_energy(geom, sigma_deg, confidence_clearance=confidence_clearance,
                                                    pixel_floor=pixel_floor)
    return energy_kernel, spatial_kernel
