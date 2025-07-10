# from gammabayes import resources_dir, power_law, haversine, power_law
# from gammabayes.utils import iterate_logspace_integration, logspace_riemann
# from gammabayes.likelihoods.irfs import IRF_LogLikelihood
# import warnings

# import numpy as np
# from gammapy.modeling.models import (
#     PowerLawSpectralModel,
#     SkyModel,
#     TemplateSpatialModel,
#     create_fermi_isotropic_diffuse_model,
# )
# from gammapy.catalog import SourceCatalogHGPS
# from gammapy.maps import Map, MapAxis, WcsGeom

# from scipy import interpolate
# from scipy.special import logsumexp
# from astropy.coordinates import SkyCoord
# from astropy import units as u

# from gammabayes.priors.core import DiscreteLogPrior, SourceFluxDiscreteLogPrior
# from gammabayes import GammaLogExposure, GammaBinning

# fermi_flux_units = 1/u.TeV/u.s/u.deg**2/(u.cm**2)
# fermi_obs_rate_units = 1/u.TeV/u.s/u.deg**2


# def construct_fermi_gaggero_flux_matrix(energy_axis: np.ndarray=None, longitudeaxis: np.ndarray=None, latitudeaxis: np.ndarray=None,
#                                    binning_geometry:GammaBinning=None,
#                                    logspace_integrator:callable=logspace_riemann):
#     """
#     Constructs the Fermi-Gaggero diffuse background event rate matrix (using Gammapy).

#     Args:
#         energy_axis (np.ndarray): Energy axis for the matrix (TeV).
#         longitudeaxis (np.ndarray): Longitude axis for the matrix (degrees).
#         latitudeaxis (np.ndarray): Latitude axis for the matrix (degrees).
#         log_exposure_map (callable): Logarithm of the exposure as a function of energy, longitude, and latitude.
#         logspace_integrator (callable, optional): Function used for integration over log space. Defaults to logspace_riemann.

#     Returns:
#         np.ndarray: A 3D array representing the observed event rate from the Fermi-Gaggero diffuse background over specified axes.
#     """
#     warnings.warn("construct_fermi_gaggero_flux_matrix will be deprecated after version 0.1.16 of GammaBayes. Please use [insert class name here]", DeprecationWarning)
    
#     if (binning_geometry is None) and (energy_axis is None):
#         raise ValueError("Either a binning geometry or the energy, longitude and latitude axes.")

#     elif not(binning_geometry is None):
#         binning_geometry = binning_geometry
#     else:
#         binning_geometry = GammaBinning(energy_axis=energy_axis, lon_axis=longitudeaxis, lat_axis=latitudeaxis) 

#     energy_axis_true = MapAxis.from_nodes(energy_axis, interp='log', name="energy_true")


#     geom = WcsGeom.create(
#         skydir=SkyCoord(0, 0, unit="deg", frame='galactic'),
#         binsz=(np.diff(longitudeaxis)[0], np.diff(latitudeaxis)[0]),
#         width=(np.ptp(longitudeaxis)+np.diff(longitudeaxis)[0], np.ptp(latitudeaxis)+np.diff(latitudeaxis)[0]),
#         frame="galactic",
#         proj="CAR",
#         axes=[energy_axis_true],
#     )


#     template_diffuse_skymap = TemplateSpatialModel.read(
#         filename=resources_dir+"/gll_iem_v06_gc.fits.gz", normalize=True
#     )

#     diffuse_iem = SkyModel(
#         spatial_model=template_diffuse_skymap,
#         spectral_model=PowerLawSpectralModel(),
#         name="diffuse-iem",
#     )

#     # Need to flip as convention goes positive to negative 
#     fermievaluated = np.flip(np.transpose(diffuse_iem.evaluate_geom(geom).to(fermi_flux_units), axes=(0,2,1)), axis=1)

#     # Normalising so I can apply the normalisation of that in Gaggero et al.
#     fermi_integral_values= logspace_integrator(logy=np.log(fermievaluated.value), x=energy_axis, axis=0)
#     fermi_integral_values = fermi_integral_values - logspace_integrator(
#         logy=logspace_integrator(
#             logy=fermi_integral_values, x=longitudeaxis, axis=0), x=latitudeaxis, axis=0)

#     # Slight change in normalisation due to the use of m^2 not cm^2 so there is a 10^4 change in the normalisation
#     fermi_gaggero = np.exp(fermi_integral_values+np.log(power_law(
#         energy_axis.to(u.TeV).value, 
#         index=-2.41, 
#         phi0=1.36*1e-8*u.Unit("TeV-1 cm-2 s-1 sr-1")
#         ).to(fermi_flux_units).value)[:, np.newaxis, np.newaxis])
    

#     return fermi_gaggero*fermi_flux_units





# class construct_log_fermi_gaggero_bkg(object):
#     """
#     Constructs and interpolates the log of the Fermi-Gaggero diffuse background model.

#     Args:
#         energy_axis (np.ndarray): Energy axis for the model (TeV).
#         longitudeaxis (np.ndarray): Longitude axis for the model (degrees).
#         latitudeaxis (np.ndarray): Latitude axis for the model (degrees).
#         logspace_integrator (callable, optional): Function used for normalization in log space. Defaults to logspace_riemann.

#     This class generates a regular grid interpolator for the Fermi-Gaggero diffuse background model,
#     allowing for log-space evaluation of the model at arbitrary points.
#     """

#     def __init__(self, binning_geometry:GammaBinning=None,):
#         """Constructs and interpolates the log of the Fermi-Gaggero diffuse background model.

#         Args:
#             energy_axis (np.ndarray): Energy axis for the model (TeV).
            
#             longitudeaxis (np.ndarray): Longitude axis for the model (degrees).
            
#             latitudeaxis (np.ndarray): Latitude axis for the model (degrees).
            
#             log_aeff (callable): Logarithm of the effective area as a function of energy, longitude, and latitude.
            
#             normalise (bool, optional): Whether to normalize the model over the specified axes. Defaults to True.
            
#             logspace_integrator (callable, optional): Function used for normalization in log space. Defaults to logspace_riemann.

#         This class generates a regular grid interpolator for the Fermi-Gaggero diffuse background model,
#         allowing for log-space evaluation of the model at arbitrary points.
#         """
#         warnings.warn("construct_log_fermi_gaggero_bkg will be deprecated after version 0.1.16 of GammaBayes. Please use [insert class name here]", DeprecationWarning)

#         self.binning_geometry = binning_geometry
            
#         energy_axis, longitudeaxis, latitudeaxis = binning_geometry.axes

#         log_fermi_diffuse = np.log(construct_fermi_gaggero_flux_matrix(energy_axis=energy_axis, 
#         longitudeaxis=longitudeaxis, latitudeaxis=latitudeaxis).to(fermi_flux_units).value)


#         # Have to interpolate actual probabilities as otherwise these maps include -inf
#         self.fermi_diffuse_flux_interpolator = interpolate.RegularGridInterpolator(
#             (*self.binning_geometry.axes,), 
#             np.exp(log_fermi_diffuse)
#             )
        



#     # Then we make a wrapper to put the result of the function in log space
#     def log_flux_func(self, energy, longitude, latitude, 
#                  spectral_parameters={}, spatial_parameters={}): 
#         """
#         Computes the log of the interpolated Fermi-Gaggero model at given energy, longitude, and latitude.

#         Args:
#             energy (float): Energy value for evaluation (TeV).
#             longitude (float): Longitude value for evaluation (degrees).
#             latitude (float): Latitude value for evaluation (degrees).
#             spectral_parameters (dict, optional): Spectral parameters, unused in this context. Defaults to an empty dict.
#             spatial_parameters (dict, optional): Spatial parameters, unused in this context. Defaults to an empty dict.

#         Returns:
#             float: The log of the Fermi-Gaggero diffuse background model at the specified location.
#         """
#         return np.log(
#         self.fermi_diffuse_flux_interpolator(
#             (energy, longitude, latitude)
#             ))