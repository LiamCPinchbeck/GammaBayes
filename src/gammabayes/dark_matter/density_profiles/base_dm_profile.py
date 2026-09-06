import numpy as np, time, torch
# import astropy.units as u
# from gammabayes.utils import logspace_riemann, logspace_simpson
from gammabayes import haversine
from gammabayes.priors.spatial_components import BaseSpatial_PriorComp
from functools import partial
from icecream import ic

class DM_Profile(BaseSpatial_PriorComp):
    """Class for dark matter density profiles and related calculations."""
    d_kpc_to_cm = torch.tensor(3.086e21)
    sr_to_deg2 = torch.tensor(3.0461741978670857e-4)


    def __init__(self, 
                 log_profile_func: callable, 
                 rho_s=torch.tensor(1.),
                 LOCAL_DENSITY  = torch.tensor(3.9e-4), #*u.Unit("TeV/cm3"), 
                 dist_to_source = torch.tensor(8.3), # u.kpc, 
                 annihilation = torch.tensor(1.),
                 angular_central_coords = torch.tensor([0,0]), #u.deg,
                 int_resolution: int = 101, 
                 *args, **kwargs
                 ):

        self._log_profile_func = log_profile_func
        self.int_resolution = int_resolution
        self.t_range = torch.linspace(0, 4, self.int_resolution)
        self.deltat = self.t_range[1] - self.t_range[0]

        _dev = self.t_range.device
        self.LOCAL_DENSITY = torch.as_tensor(LOCAL_DENSITY).to(_dev)
        self.DISTANCE      = torch.as_tensor(dist_to_source).to(_dev)
        self.annihilation  = torch.as_tensor(annihilation).to(_dev)
        self.rho_s         = torch.as_tensor(rho_s).to(_dev)
        self.angular_central_coords = torch.as_tensor(angular_central_coords).to(_dev)
        self.d_kpc_to_cm   = self.d_kpc_to_cm.to(_dev)
        self.sr_to_deg2    = self.sr_to_deg2.to(_dev)

        self.log_integral_constants = torch.log(self.deltat * self.d_kpc_to_cm * self.sr_to_deg2)
        self.log_integral_constants = self.log_integral_constants.to(self.DISTANCE.device)

        self.log_profile_func = self.scale_density_profile_to_refs(
            ref_density= self.LOCAL_DENSITY, ref_density_radius = self.DISTANCE, 
            rho_s=rho_s, log_profile_func = self._log_profile_func, *args, **kwargs)


        super().__init__(logfunc=self.logdiffJ)


    @staticmethod
    def calc_new_scale_density_to_refs(ref_density, ref_density_radius, rho_s, log_profile_func, *args, **kwargs):
        log_density = log_profile_func(ref_density_radius, rho_s=rho_s, *args,**kwargs)
        density_ref_ratio = ref_density/log_density.exp()
        rho_s = rho_s*density_ref_ratio
        return rho_s


    @staticmethod
    def scale_density_profile_to_refs(ref_density, ref_density_radius, rho_s, log_profile_func, *args, **kwargs):

        rho_s = DM_Profile.calc_new_scale_density_to_refs(
            ref_density=ref_density, ref_density_radius=ref_density_radius, log_profile_func=log_profile_func,
            rho_s=rho_s, *args, **kwargs)

        __log_profile_func = partial(log_profile_func, rho_s=rho_s, *args, **kwargs)

        return __log_profile_func


    def logdiffJ(self, lonlatgrid, **kwargs) :

        lon_grid = lonlatgrid[..., 0]
        lat_grid = lonlatgrid[..., 1]
        
        angular_offset = haversine(lon_grid.flatten(), 
                                   lat_grid.flatten(), 
                                   self.angular_central_coords[0], 
                                   self.angular_central_coords[1],)

        return self._log_diffJ_from_offset(theta=angular_offset, kwargs=kwargs).reshape(lat_grid.shape) # No point unpacking, packing and the re-unpacking the kwargs


    def _log_diffJ_from_offset(self, theta, kwargs):
        # Convert to radians
        theta_rad = torch.deg2rad(theta)

        # Figure out ds/dt for line integral
        dsdt = self.DISTANCE/torch.abs(torch.cos(theta_rad))

        # Figure out direction vectors for line integrals _\|/_
        vel_dirvecs = torch.stack([self.DISTANCE*torch.tan(theta_rad), self.DISTANCE*torch.ones_like(theta_rad)], dim=0)

        # Calculate positions along the lines of sight:  p_ref + t*dir(p)
        positions = torch.stack([torch.zeros_like(self.DISTANCE), -self.DISTANCE])[:, None, None] + self.t_range[None, None, :]*vel_dirvecs[:, :, None]

        # Calculate the radii of the positions along the lines of sight
        radii = torch.sqrt((positions**2).sum(dim=(0)))

        # Re-anchor rho(r_sun) = LOCAL_DENSITY for the CURRENT shape kwargs (r_s, gamma, ...).
        # Exactly 0 when kwargs empty (init scaling), so back-compatible.
        log_norm_corr = torch.log(self.LOCAL_DENSITY) - self.log_profile_func(self.DISTANCE, **kwargs)
        log_densities_to_integrate = (1.+self.annihilation)*(self.log_profile_func(radii, **kwargs) + log_norm_corr)
        
        # Integrate and return
        return torch.logsumexp(log_densities_to_integrate + torch.log(dsdt[:, None]), dim=1) + self.log_integral_constants




