import torch
from torch.distributions import Normal, Categorical


def make_normal_kernel_with_std_and_bins(linear_axis, std, confidence_clearance=5):
    """1-D Gaussian kernel sampled at the spacing of ``linear_axis``, ``confidence_clearance`` sigmas wide
    (odd number of taps, not normalised).  Used by the legacy :class:`LogSumExpConv3D` operator."""

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
    """Separable 3-D Gaussian IRF kernel ``(K_E, K_lon, K_lat)`` for the legacy ``LogSumExpConv3D``.

    ``lon_std`` / ``lat_std`` are the PSF sigma in degrees.  ``log10energy_std`` is divided by
    log10(e) before use, so the effective energy width is ln(10) times the value passed; this is kept
    for reproducibility of 0.x results.  New analyses should use
    :func:`gammabayes.utils.irf_forward.make_forward_irf_kernels`, which uses the width as given.
    """

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
    """Event-level IRF smearing for simulations: samples an energy / longitude / latitude offset per event
    from separable Gaussian kernels on the measured geometry and redistributes a counts cube accordingly
    (``sample_kernel``).  Events smeared outside the grid are dropped."""

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

        self.num_energy_kernel_bins = int(torch.ceil(torch.as_tensor(self.log_energy_sigma*kernel_sigmas/self.log10_energy_diff)))
        self.num_lon_kernel_bins = int(torch.ceil(torch.as_tensor(self.lon_sigma*kernel_sigmas/self.lon_diff)))
        self.num_lat_kernel_bins = int(torch.ceil(torch.as_tensor(self.lat_sigma*kernel_sigmas/self.lat_diff)))

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
        """Smear an integer counts cube (E, lon, lat) through the kernel by sampling one kernel offset
        per event; events that leave the grid are dropped."""
        counts = torch.as_tensor(true_value_grid).round().to(torch.int64)
        shape = counts.shape
        num_samples = int(counts.sum())
        out = torch.zeros(shape, dtype=torch.as_tensor(true_value_grid).dtype, device=counts.device)
        if num_samples == 0:
            return out
        energy_indices, lon_indices, lat_indices = self._sample_indices(num_samples)
        flat = counts.flatten()
        voxel = torch.repeat_interleave(torch.arange(flat.numel(), device=counts.device), flat)   # (N,) in voxel order
        e0, l0, b0 = torch.unravel_index(voxel, shape)
        e = e0 + energy_indices.to(counts.device); l = l0 + lon_indices.to(counts.device); b = b0 + lat_indices.to(counts.device)
        ok = (e >= 0) & (e < shape[0]) & (l >= 0) & (l < shape[1]) & (b >= 0) & (b < shape[2])
        out.index_put_((e[ok], l[ok], b[ok]), torch.ones(int(ok.sum()), dtype=out.dtype, device=out.device), accumulate=True)
        return out


# ---------------------------------------------------------------------------
# Energy-calibrated PSF kernels
# ---------------------------------------------------------------------------
def make_spatial_kernel_per_energy(geom, sigma_deg, confidence_clearance=5, pixel_floor=False):
    """Per-true-energy 2-D Gaussian PSF kernels on a common tap grid.

    sigma_deg: (E,) tensor of Gaussian widths [deg] (``IRFExtractor.psf_sigma_vs_energy``).
    The tap grid is sized by the largest sigma (``confidence_clearance`` sigmas,
    odd number of taps); planes with a narrower PSF concentrate their weight near the
    centre of the same grid.  Each plane is normalised to unit sum.  Returns (E, K_lon, K_lat).

    pixel_floor adds Delta^2/12 per axis in quadrature; apply it either here or in
    ``psf_sigma_vs_energy``, not in both.
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
