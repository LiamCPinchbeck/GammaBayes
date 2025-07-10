# Necessary imports for the script functionality
from gammabayes import resources_dir, haversine, GammaBinning, GammaLogExposure
from gammabayes.utils import iterate_logspace_integration
from gammabayes.likelihoods.irfs import IRF_LogLikelihood
from gammabayes.priors.core import DiscreteLogPrior, SourceFluxDiscreteLogPrior


import numpy as np, logging
from scipy import interpolate
from scipy.special import logsumexp
from astropy.coordinates import SkyCoord
from astropy import units as u

from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
)
from gammapy.catalog import SourceCatalogHGPS

hess_flux_units = 1/u.TeV/u.s/u.deg**2/(u.cm**2)
hess_obs_rate_units = 1/u.TeV/u.deg**2


def construct_hess_flux_matrix(binning_geometry:GammaBinning=None,
        energy_axis: np.ndarray=None, longitudeaxis: np.ndarray=None, latitudeaxis: np.ndarray=None,):
    """
    Constructs a map of HESS source fluxes based on the HGPS catalog.

    Args:
        energy_axis (np.ndarray): Energy axis for the map (TeV).
        longitudeaxis (np.ndarray): Longitude axis for the map (degrees).
        latitudeaxis (np.ndarray): Latitude axis for the map (degrees).
        log_aeff (callable): Logarithm of the effective area as a function of energy, longitude, and latitude.
        aeff_unit (optional): Units for effective area function.

    Returns:
        np.ndarray: A 3D array representing the event rate from HESS catalog sources over the specified energy,
                    longitude, and latitude axes.
    """

    if binning_geometry is None:
        binning_geometry = GammaBinning(energy_axis=energy_axis, lon_axis=longitudeaxis, lat_axis=latitudeaxis)
    else:
        binning_geometry = binning_geometry
        

    hess_catalog = SourceCatalogHGPS(resources_dir+"/hgps_catalog_v1.fits.gz")

    hess_models = hess_catalog.to_models()

    centre_spatial = binning_geometry.spatial_centre

    energy_axis_true_nodes = MapAxis.from_nodes(binning_geometry.energy_axis, interp='log', name="energy_true")

    HESSgeom = WcsGeom.create(
        skydir=SkyCoord(centre_spatial[0], centre_spatial[1], unit="deg", frame='galactic'),
        binsz=(np.diff(binning_geometry.lon_axis)[0], np.diff(binning_geometry.lat_axis)[0],),
        width=(np.ptp(binning_geometry.lon_axis)+np.diff(binning_geometry.lon_axis)[0], 
               np.ptp(binning_geometry.lat_axis)+np.diff(binning_geometry.lat_axis)[0]),
        frame="galactic",
        proj="CAR",
        axes=[energy_axis_true_nodes],
    )

    HESSmap = Map.from_geom(HESSgeom)

    
    count=0 # To keep track of the number of sources satisfying the conditions

    full_hess_flux = 0 #Extracting the flux values for the sources along the axes previously defined

    for idx, model in enumerate(hess_models):
        
        # We will store the longitude 'l' and latitude 'b' values of the source
        temp_l_value = model.spatial_model.position.l.degree*u.deg
        temp_b_value = model.spatial_model.position.b.degree*u.deg
        
        # The unit convention for the longitude works as 357, 358, 359, 0, 1, 2
            # we will make it so that it's -3, -2, -1, 0, 1, 2
        if temp_l_value>180*u.deg:
            temp_l_value=temp_l_value-360*u.deg

            

        lon_include_condition = np.logical_and(temp_l_value<np.max(binning_geometry.lon_axis), temp_l_value>np.min(binning_geometry.lon_axis))
        lat_include_condition = np.logical_and(temp_b_value<np.max(binning_geometry.lat_axis), temp_b_value>np.min(binning_geometry.lat_axis))
            


        # If the location of the source satisfies the criteria above then we start to store it's flux values
        if  lon_include_condition and lat_include_condition:
            
            # We extract the flux by assigning it to the HESSmap object
            HESSmap.quantity = model.evaluate_geom(HESSmap.geom)
            # For future debuggers, HESSmap.quantity should have units of 1/TeV s sr cm^2

            
            # We can then extract the actual values by referencing the data of this object
            startdata = HESSmap.data
            
            # Removing nan values, '~' is a shortcut for 'not'
            data = startdata[~np.isnan(startdata)]
            
            # If statement to catch sources that have no obervable flux within the region
            if data.size!=0:
                
                # Making sure the data shape is consistent
                data = data.reshape(startdata.shape)
                
                # Adding the flux from this model to the total flux
                full_hess_flux+=data
                    
                
            count+=1
            
    # Transposing the longitude and latitude values such that the order of indices goes [logenergy_index, longitude_index, latitude_index]
    full_hess_flux = np.transpose(full_hess_flux, axes=(0,2,1))
    full_hess_flux = np.flip(full_hess_flux, axis=1)
    
    # hessgeom = HESSmap.geom

    # # Extracting the longitudevalues
    # hesslonvals = hessgeom.get_coord().lon[0][0]

    # # Reversing the order as the order is flipped by convention
    # hesslonvals = hesslonvals[::-1]

    # # Again converting the longitude axis from 357,358,359,0,1,2,3 to -3,-2,-1,0,1,2,3
    # hesslonvals[hesslonvals>180*u.deg] = hesslonvals[hesslonvals>180*u.deg]-360*u.deg
    

    return (full_hess_flux*HESSmap.quantity.unit).to(hess_flux_units)
                


class construct_hess_source_map_interpolation(object):
    """
    Provides interpolated values from the HESS source map for given energy, longitude, and latitude points.

    Args:
        energy_axis (np.ndarray): Energy axis for the interpolation (TeV).
        longitudeaxis (np.ndarray): Longitude axis for the interpolation (degrees).
        latitudeaxis (np.ndarray): Latitude axis for the interpolation (degrees).
        log_aeff (callable): Function to calculate the log of the effective area.
        normalise (bool, optional): Whether to normalise the source map. Defaults to True.
        iterate_logspace_integrator (callable, optional): Integrator function for normalisation in log space. Defaults to iterate_logspace_integration.
        aeff_unit (optional): Units for effective area function. Defaults to u.cm**2.
    """
    
    def __init__(self,
                 binning_geometry:GammaBinning=None,):
        """Provides interpolated values from the HESS source map for given energy, longitude, and latitude points.

        Args:
            'To be written'
        This class constructs an interpolation function for the HESS source map, allowing for the evaluation of the
        map at arbitrary points within the defined axes.
        """

        self.binning_geometry = binning_geometry

        log_astro_sourcemap = np.log(construct_hess_flux_matrix(binning_geometry=self.binning_geometry,).to(hess_flux_units).value)



        # Have to interpolate actual probabilities as otherwise these maps include -inf
        self.hess_grid_interpolator = interpolate.RegularGridInterpolator(
            (*self.binning_geometry.axes,), 
            np.exp(log_astro_sourcemap),
            bounds_error=False, fill_value=0)

    # Then we make a wrapper to put the result of the function in log space
    def log_func(self, energy, longitude, latitude, 
                       spectral_parameters={}, spatial_parameters={}):
        """
        Computes the log of interpolated values from the HESS source map at given points.

        Args:
            energy (float): Energy value for the interpolation (TeV).
            longitude (float): Longitude value for the interpolation (degrees).
            latitude (float): Latitude value for the interpolation (degrees).
            spectral_parameters (dict, optional): Spectral parameters for the model. Defaults to an empty dict.
            spatial_parameters (dict, optional): Spatial parameters for the model. Defaults to an empty dict.

        Returns:
            float: Log of the interpolated value from the HESS source map at the specified point.
        """
        return np.log(self.hess_grid_interpolator((energy, longitude, latitude)))


class HESSCatalogueSources_Prior(SourceFluxDiscreteLogPrior):
    """
    Defines a prior based on HESS catalogue sources over specified energy, longitude, and latitude axes.

    Args:
        energy_axis (np.ndarray): Energy axis for the prior (TeV).
        longitudeaxis (np.ndarray): Longitude axis for the prior (degrees).
        latitudeaxis (np.ndarray): Latitude axis for the prior (degrees).
        irf (IRF_LogLikelihood): Instrument Response Function log likelihood instance.
        normalise (bool, optional): Whether to normalise the prior. Defaults to True.
        iterate_logspace_integrator (callable, optional): Function for integration over log space for normalisation. Defaults to iterate_logspace_integration.
    """



    
    def __init__(self, 
                 axes: list[np.ndarray[u.Quantity]]| tuple[np.ndarray[u.Quantity]]=None,
                 binning_geometry: GammaBinning = None,
                 *args, **kwargs):
        """Defines a prior based on HESS catalogue sources over specified energy, longitude, and latitude axes.

        Args:
            energy_axis (np.ndarray): Energy axis for the prior (TeV).
            
            longitudeaxis (np.ndarray): Longitude axis for the prior (degrees).
            
            latitudeaxis (np.ndarray): Latitude axis for the prior (degrees).
            
            irf (IRF_LogLikelihood): Instrument Response Function log likelihood instance.
            
            normalise (bool, optional): Whether to normalise the prior. Defaults to True.
            
            iterate_logspace_integrator (callable, optional): Function for integration over log space for normalisation. Defaults to iterate_logspace_integration.
        
        This class utilizes an interpolation of the HESS source map to define a prior distribution for Bayesian analysis,
        integrating the map's flux values with the instrument's effective area.
        """


        self._create_geometry(axes=axes, binning_geometry=binning_geometry)




        self.log_hess_class_instance = construct_hess_source_map_interpolation(binning_geometry=self.binning_geometry,)
        
        
        super().__init__(
            name='HESSCatalogueSources_Prior',
            axes=self.binning_geometry.axes,
            binning_geometry=self.binning_geometry,
            log_flux_function=self.log_hess_class_instance.log_func, 
            *args, **kwargs
        )

        self.unit = hess_obs_rate_units


