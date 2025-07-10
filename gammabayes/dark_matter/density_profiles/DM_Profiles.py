import numpy as np
import astropy.units as u


import numpy as np
import astropy.units as u
from gammabayes.utils import logspace_riemann
from gammabayes import haversine, update_with_defaults
from .base_dm_profile import DM_Profile
import time




class Einasto_Profile(DM_Profile):
    """Einasto dark matter density profile."""

    @staticmethod
    def log_density_profile_func(radius, r_s, alpha, rho_s):
        """
        Computes the logarithm of the Einasto density profile.

        Args:
            radius (tensor): Radial distance.
            r_s (tensor): Scale radius.
            alpha (tensor): Einasto shape parameter.
            rho_s (tensor): Scale density.
        """

        rr = radius / r_s

        exponent = -(2 / alpha) * (rr**alpha - 1)

        return torch.log(rho_s) + exponent




    def __init__(self, 
                    alpha = torch.tensor(0.17), 
                    rho_s = torch.tensor(0.001), # u.Unit(TeV / cm3)", 
                    r_s= torch.tensor(20.), # u.Unit("kpc"), 
                    *args, **kwargs):
        """
        Initializes the Einasto_Profile class.

        Args:
            alpha (tensor, optional): Default Einasto shape parameter. Defaults to 0.17.
            rho_s (tensor, optional): Default scale density. Defaults to 1. * u.Unit("TeV / cm3").
            r_s (tensor, optional): Default scale radius. Defaults to 20.00 * u.Unit("kpc").
        """


        super().__init__(
            log_profile_func=self._log_profile_func, 
            rho_s=rho_s,
            r_s=r_s,
            *args, **kwargs
        )



# class GNFW_Profile(DM_Profile):
#     """Generalized NFW (GNFW) or Zhao dark matter density profile."""

#     @staticmethod
#     def log_profile(radius: float | np.ndarray , 
#                          r_s: float, 
#                          alpha: float, 
#                          beta: float, 
#                          gamma: float, 
#                          rho_s: float):
#         """
#         Computes the logarithm of the GNFW density profile.

#         Args:
#             radius (float | np.ndarray): Radial distance.
#             r_s (float): Scale radius.
#             alpha (float): alpha parameter.
#             beta (float): beta parameter.
#             gamma (float): gamma parameter.
#             rho_s (float): Scale density.

#         Returns:
#             float | np.ndarray: Logarithm of the density profile.
#         """
        
#         radial_unit = u.kpc

#         radius = radius.to(radial_unit).value if hasattr(radius, "unit") else radius

#         r_s =  r_s.to(radial_unit).value if hasattr(radius, "unit") else radius

#         rho_s =  rho_s.to("TeV / cm3").value if hasattr(rho_s, "unit") else radius

#         rr = radius / r_s

#         result = np.log(rho_s) - gamma*np.log(rr) + ((gamma - beta) * alpha) * np.log(1 + rr **  1/alpha)

#         return result
    
#     def __init__(self, 
#                  default_alpha: float | int = 1, 
#                  default_beta: float | int = 3, 
#                  default_gamma: float | int = 1, 
#                  default_rho_s: float = 0.001* u.Unit("TeV / cm3"), 
#                  default_r_s: float = 24.42* u.Unit("kpc"),
#                 *args, **kwargs):
#         """
#         Initializes the GNFW_Profile class.

#         Args:
#             default_alpha (float | int, optional): Default alpha parameter. Defaults to 1.
#             default_beta (float | int, optional): Default beta parameter. Defaults to 3.
#             default_gamma (float | int, optional): Default gamma parameter. Defaults to 1.
#             default_rho_s (float, optional): Default scale density. Defaults to 0.001 * u.Unit("TeV / cm3").
#             default_r_s (float, optional): Default scale radius. Defaults to 24.42 * u.Unit("kpc").
#         """
#         super().__init__(
#             log_profile_func=self.log_profile, 
#             kwd_profile_default_vals = {'r_s': default_r_s, 
#                                         'alpha': default_alpha, 
#                                         'beta': default_beta,
#                                         'gamma': default_gamma,
#                                         'rho_s':default_rho_s},
#             gammapy_profile_class=Gammapy_GNFWProfile,
#             *args, **kwargs
#         )



# class Burkert_Profile(DM_Profile):
#     """Burkert dark matter density profile."""

#     @staticmethod
#     def log_profile(radius, r_s, rho_s):
#         """
#         Computes the logarithm of the Burkert density profile.

#         Args:
#             radius (float | np.ndarray): Radial distance.
#             r_s (float): Scale radius.
#             rho_s (float): Scale density.

#         Returns:
#             float | np.ndarray: Logarithm of the density profile.
#         """
#         radius_unit = radius.unit
#         radius = radius.to(radius_unit).value
#         r_s = r_s.to(radius_unit).value
#         rr = radius / r_s

#         return np.log(rho_s.to("TeV / cm3").value) - (np.log(1 + rr) + np.log(1 + rr**2))


    
#     def __init__(self, 
#                  default_rho_s: float = 0.001* u.Unit("TeV / cm3"), 
#                  default_r_s: float = 12.67* u.Unit("kpc"), 
#                  *args, **kwargs):
#         """
#         Initializes the Burkert_Profile class.

#         Args:
#             default_rho_s (float, optional): Default scale density. Defaults to 0.001 * u.Unit("TeV / cm3").
#             default_r_s (float, optional): Default scale radius. Defaults to 12.67 * u.Unit("kpc").
#         """
#         super().__init__(
#             log_profile_func=self.log_profile, 
#             kwd_profile_default_vals = {'r_s': default_r_s, 
#                                         'rho_s':default_rho_s},
#             gammapy_profile_class=Gammapy_BurkertProfile,
#             *args, **kwargs
#         )
 


# class Moore_Profile(DM_Profile):
#     """Moore dark matter density profile."""

#     @staticmethod
#     def log_profile(radius, r_s, rho_s):
#         """
#         Computes the logarithm of the Moore density profile.

#         Args:
#             radius (float | np.ndarray): Radial distance.
#             r_s (float): Scale radius.
#             rho_s (float): Scale density.

#         Returns:
#             float | np.ndarray: Logarithm of the density profile.
#         """
#         rr = radius / r_s
#         return np.log(rho_s) - 1.16*np.log(rr)  - 1.84* np.log(1 + rr)
    
#     def __init__(self, 
#                  default_rho_s: float = 0.001* u.Unit("TeV / cm3"), 
#                  default_r_s: float = 30.28* u.Unit("kpc"), 
#                  *args, **kwargs):
#         """
#         Initializes the Moore_Profile class.

#         Args:
#             default_rho_s (float, optional): Default scale density. Defaults to 0.001 * u.Unit("TeV / cm3").
#             default_r_s (float, optional): Default scale radius. Defaults to 30.28 * u.Unit("kpc").
#         """
#         super().__init__(
#             log_profile_func=self.log_profile, 
#             kwd_profile_default_vals = {'r_s': default_r_s, 
#                                         'rho_s':default_rho_s},
#             gammapy_profile_class=Gammapy_MooreProfile,
#             *args, **kwargs
#         )
