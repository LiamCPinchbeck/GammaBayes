import numpy as np
import astropy.units as u
# from gammabayes.utils import logspace_riemann, logspace_simpson
from gammabayes import haversine, update_with_defaults
from scipy import special
import time


from gammapy.astro.darkmatter.profiles import (
    DMProfile as Gammapy_DMProfile,
    EinastoProfile as Gammapy_EinastoProfile,
    NFWProfile as Gammapy_NFWProfile,
    BurkertProfile as Gammapy_BurkertProfile,
    MooreProfile as Gammapy_MooreProfile,
    IsothermalProfile as Gammapy_IsothermalProfile,
)



from gammapy.utils.integrate import trapz_loglog

# from gammabayes.utils.integration import logspace_trapz
from icecream import ic


class DM_Profile(object):
    """Class for dark matter density profiles and related calculations."""

    def scale_density_profile(self, density, distance, 
                              **kwargs):
        """
        Scales the density profile to match a given density at a specific distance.

        Args:
            density (Quantity): The density to match.
            distance (Quantity): The distance at which to match the density.
        """
        scale = density / (np.exp(self.log_profile(distance, **kwargs))*u.Unit("TeV/cm3"))
        self.default_rho_s *= scale.to("")

    def __init__(self, log_profile_func: callable, 
                 LOCAL_DENSITY  = 0.39*u.Unit("GeV/cm3"), 
                 dist_to_source = 8.5*u.kpc, 
                 annihilation = 1,
                 default_rho_s = 1 * u.Unit("GeV / cm3"), 
                 default_r_s = 20.* u.Unit("kpc"), 
                 angular_central_coords = np.array([0,0])*u.deg,
                 kwd_profile_default_vals = {},
                 gammapy_profile_class=Gammapy_EinastoProfile,
                 diffJ_units = u.Unit("TeV2 cm-5 deg-2"),
                 diffD_units = u.Unit("TeV cm-2 deg-2")
                 ):
        """
        Initializes the DM_Profile class with specified parameters.

        Args:
            log_profile_func (callable): Function to compute the logarithm of the density profile.
            LOCAL_DENSITY (Quantity, optional): Local dark matter density. Defaults to 0.39 * u.Unit("GeV/cm3").
            dist_to_source (Quantity, optional): Distance to the source. Defaults to 8.33 * u.kpc.
            annihilation (int, optional): Annihilation flag. Defaults to 1 for annihilation.
            default_rho_s (Quantity, optional): Default scale density. Defaults to 0.001 * u.Unit("TeV/cm3").
            default_r_s (Quantity, optional): Default scale radius. Defaults to 28.44 * u.Unit("kpc").
            angular_central_coords (Quantity, optional): Central coordinates. Defaults to np.array([0, 0]) * u.deg.
            kwd_profile_default_vals (dict, optional): Default keyword parameters for the profile function.
            gammapy_profile_class (class, optional): Gammapy profile class. Defaults to EinastoProfile.
            diffJ_units (Quantity, optional): Units for differential J factor. Defaults to u.Unit("TeV2 cm-5 deg-2").
            diffD_units (Quantity, optional): Units for differential D factor. Defaults to u.Unit("TeV cm-2 deg-2").
        """
        self.log_profile_func           = log_profile_func
        self.LOCAL_DENSITY              = LOCAL_DENSITY
        self.DISTANCE                   = dist_to_source
        self.annihilation               = annihilation
        self.kwd_profile_default_vals   = kwd_profile_default_vals
        self.default_r_s                = default_r_s
        self.default_rho_s              = default_rho_s
        self.angular_central_coords     = angular_central_coords
        self.scale_density_profile(self.LOCAL_DENSITY, self.DISTANCE, **kwd_profile_default_vals)

        self.diffJ_units = diffJ_units
        self.diffD_units = diffD_units

        ########################################
        ########################################
        # Gammapy stuff
        self.gammapy_profile            = gammapy_profile_class(
            r_s = self.default_r_s,
            rho_s=self.default_rho_s)
        # Bit sneaky, but I don't want my class to have two different "distance"
            # attributes
        self.gammapy_profile.distance = self.DISTANCE
        self.gammapy_profile.scale_to_local_density()

    def __call__(self, *args, **kwargs) -> float | np.ndarray :
        """
        Enables using the DM_Profile instance as a callable, computing the log differential J factor.

        Returns:
            float | np.ndarray: The computed log differential J factor.
        """
        return self.compute_logdifferential_jfactor(*args, **kwargs)
    
    def log_density(self, *args, **kwargs):
        """
        Computes the logarithm of the density profile.

        Returns:
            float | np.ndarray: The logarithm of the density.
        """
        update_with_defaults(kwargs, self.kwd_profile_default_vals)

        return self.log_profile_func(*args, **kwargs)

    def compute_logdifferential_jfactor(self, 
                                        longitude, 
                                        latitude, ndecade:int|float = 1e4, 
                                        kwd_parameters:dict={}):
        """
        Computes the logarithm of the differential J factor.

        Args:
            longitude (Quantity): Longitude values.
            latitude (Quantity): Latitude values.
            ndecade (int | float, optional): Number of points per decade for integration. Defaults to 1e4.
            kwd_parameters (dict, optional): Additional parameters for the profile function.

        Returns:
            float | np.ndarray: The computed log differential J factor.
        """
        separation = haversine(longitude, latitude, *self.angular_central_coords).to(u.rad)

        rmin = u.Quantity(
            value=np.tan(separation) * self.gammapy_profile.distance, unit=self.gammapy_profile.distance.unit
        )

        rmax = self.gammapy_profile.distance
        val = [
            (
                2
                * self.integral(
                    _.value * u.kpc,
                    rmax,
                    np.arctan(_.value / self.gammapy_profile.distance.value),
                    ndecade,
                    kwd_parameters={Key:keyval for Key, keyval in zip(kwd_parameters.keys(), kwdvals)},
                )
                + self.integral(
                    self.gammapy_profile.distance,
                    5 * rmax,
                    np.arctan(_.value / self.gammapy_profile.distance.value),
                    ndecade,
                    kwd_parameters={Key:keyval for Key, keyval in zip(kwd_parameters.keys(), kwdvals)},

                )
            )
            for _, *kwdvals in zip(rmin.ravel(), *kwd_parameters.values())
        ]
        integral_unit = self.diffJ_units if self.annihilation else self.diffD_units
        jfact = (u.Quantity(val).reshape(rmin.shape)/u.steradian).to(integral_unit)
        
        return np.log(jfact.value)

    
    def _radius(self, t: float | np.ndarray, 
                angular_offset: float | np.ndarray, 
                distance: float | np.ndarray) -> float | np.ndarray :
        """
        Computes the radius for given parameters.

        Args:
            t (float | np.ndarray): Parameter t.
            angular_offset (Quantity): Angular offset.
            distance (Quantity): Distance.

        Returns:
            float | np.ndarray: The computed radius.
        """
        
        angular_offset = angular_offset.to(u.rad)

        t_mesh, offset_mesh = np.meshgrid(t, angular_offset, indexing='ij')

        costheta = np.cos(offset_mesh)
        sintheta = np.sin(offset_mesh)
        inside = t_mesh**2*costheta**2*sintheta**2 + t_mesh**2*costheta**2 - 2*t_mesh*costheta + 1
        returnval = distance*np.sqrt(inside)

        return returnval


    def logdiffJ(self, longitude: float | np.ndarray, 
                 latitude: float | np.ndarray, 
              int_resolution: int = 1001, 
              integration_method: callable = logspace_riemann, 
              kwd_parameters = {}) -> float | np.ndarray :
        """
        Computes the logarithm of the differential J factor.

        Args:
            longitude (float | np.ndarray): Longitude values.
            latitude (float | np.ndarray): Latitude values.
            int_resolution (int, optional): Integration resolution. Defaults to 1001.
            integration_method (callable, optional): Integration method. Defaults to logspace_riemann.
            kwd_parameters (dict, optional): Additional parameters for the profile function.

        Returns:
            float | np.ndarray: The computed log differential J factor.
        """
        
        angular_offset = haversine(longitude, 
                                   latitude, 
                                   self.angular_central_coords[0], 
                                   self.angular_central_coords[1],)
        
        update_with_defaults(kwd_parameters, self.kwd_profile_default_vals)

        t = np.linspace(0, 6, int_resolution)
        logy= (1+self.annihilation)*self.log_profile_func(self._radius(t, angular_offset, self.DISTANCE),  **kwd_parameters)

        logintegral = integration_method(
            logy=logy,
            x=t, 
            axis=0)
        

        
        return logintegral+np.log(self.DISTANCE.to("cm"))+np.log(np.cos(angular_offset.to("rad"))) 
    

    def mesh_efficient_logfunc(self, longitude, latitude, kwd_parameters={}, *args, **kwargs) -> float | np.ndarray :
        """
        Computes the log differential J factor efficiently using a mesh grid.

        Args:
            longitude (Quantity): Longitude values.
            latitude (Quantity): Latitude values.
            kwd_parameters (dict, optional): Additional parameters for the profile function.

        Returns:
            float | np.ndarray: The computed log differential J factor.
        """

        longitude_units = longitude.unit
        latitude_units = latitude.unit

        parameter_meshes = np.meshgrid(longitude, latitude, *kwd_parameters.values(), indexing='ij')
        parameter_values_flattened_meshes = np.asarray([mesh.flatten() for mesh in parameter_meshes])


        return self(
            longitude=parameter_values_flattened_meshes[0]*longitude_units, 
            latitude=parameter_values_flattened_meshes[1]*latitude_units, 
            kwd_parameters = {param_key: parameter_values_flattened_meshes[2+idx] for idx, param_key in enumerate(kwd_parameters)},
            *args, 
            **kwargs
            ).reshape(parameter_meshes[0].shape)
    

    def _eval_substitution(self, radius, separation, squared, kwd_parameters=None):
        """Density at given radius together with the substitution part. Taken from Gammapy to change how rmin is handled in integrate_spectrum_separation."""
        exponent = 2 if squared else 1
        if kwd_parameters is None:
            kwd_parameters = {}

        update_with_defaults(kwd_parameters, self.kwd_profile_default_vals)

        return (
            self.gammapy_profile.evaluate(radius, **kwd_parameters) ** exponent
            * radius
            / np.sqrt(radius**2 - (self.gammapy_profile.DISTANCE_GC * np.sin(separation)) ** 2)
        )

    def integral(self, rmin, rmax, separation, ndecade, squared=True, kwd_parameters=None):
        r"""Integrate dark matter profile numerically. Taken from Gammapy to change how rmin is handled in integrate_spectrum_separation.

        .. math::
            F(r_{min}, r_{max}) = \int_{r_{min}}^{r_{max}}\rho(r)^\gamma dr \\
            \gamma = 2 \text{for annihilation} \\
            \gamma = 1 \text{for decay}

        Parameters
        ----------
        rmin, rmax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        separation : `~numpy.ndarray`
            Separation angle in radians.
        ndecade : int, optional
            Number of grid points per decade used for the integration.
            Default is 10000.
        squared : bool, optional
            Square the profile before integration.
            Default is True.
        """
        if kwd_parameters is None:
            kwd_parameters = {}

        integral = self.integrate_spectrum_separation(
            self._eval_substitution, rmin, rmax, separation, ndecade, squared, kwd_parameters=kwd_parameters,
        )
        inegral_unit = u.Unit("GeV2 cm-5") if squared else u.Unit("GeV cm-2")
        return integral.to(inegral_unit)

    def integrate_spectrum_separation(
        self, func, xmin, xmax, separation, ndecade, squared=True, kwd_parameters=None
    ):
        """Squared dark matter profile integral. Taken from Gammapy for minor changes to how xmin is handled.

        Parameters
        ----------
        xmin, xmax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        separation : `~numpy.ndarray`
            Separation angle in radians.
        ndecade : int
            Number of grid points per decade used for the integration.
        squared : bool
            Square the profile before integration.
            Default is True.
        """
        if kwd_parameters is None:
            kwd_parameters = {}
        
        unit = xmin.unit
        xmin = xmin.value
        xmax = xmax.to_value(unit)
        integral_addition = 0
        if np.isclose(xmin, 0, atol=1e-10):
            inbetween_xs = np.linspace(xmin, 1e-10, int(round(ndecade)))
            inbetween_func_eval = func(inbetween_xs*unit, separation, squared, kwd_parameters=kwd_parameters)
            inbetween_func_eval = inbetween_func_eval.value
            integral_addition = np.exp(logspace_trapz(logy=np.log(inbetween_func_eval), x=inbetween_xs))
            if np.isnan(integral_addition):
                integral_addition = 0
            xmin = 1e-10


        logmin = np.log10(xmin)
        logmax = np.log10(xmax)
        n = np.int32((logmax - logmin) * ndecade)
        x = np.logspace(logmin, logmax, n) * unit
        y = func(x, separation, squared, kwd_parameters=kwd_parameters)
        val = trapz_loglog(y, x)
        integral_addition*=val.unit
        return (val+integral_addition).sum()
