from gammabayes import resources_dir, haversine
from gammabayes.priors.spectral_components import PowerLaw
from gammabayes.priors.spatial_components import BaseSpatial_PriorComp
from gammabayes.utils import iterate_logspace_integration, logspace_riemann
from gammabayes.likelihoods.irfs import IRF_LogLikelihood


import numpy as np
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
    create_fermi_isotropic_diffuse_model,
)
from gammapy.catalog import SourceCatalogHGPS
from gammapy.maps import Map, MapAxis, WcsGeom

from scipy import interpolate
from scipy.special import logsumexp
from astropy.coordinates import SkyCoord
from astropy import units as u

from gammabayes.priors.core import DiscreteLogPrior, SourceFluxDiscreteLogPrior, TwoCompFluxPrior
from gammabayes import GammaLogExposure, GammaBinning

fermi_flux_units = 1/u.TeV/u.s/u.deg**2/(u.cm**2)
fermi_obs_rate_units = 1/u.TeV/u.s/u.deg**2

template_diffuse_skymap = TemplateSpatialModel.read(
    filename=resources_dir+"/gll_iem_v06_gc.fits.gz", normalize=True
)


class FermiIEM_Morphology(BaseSpatial_PriorComp):

    units = 1/u.deg**2
    
    @staticmethod
    def fermi_iem_log_template(lon, lat, *args, **kwargs):
        return np.log(template_diffuse_skymap(lon, lat, *args, **kwargs).to(FermiIEM_Morphology.units).value)


    def __init__(self, *args, **kwargs):


        super().__init__(logfunc = self.fermi_iem_log_template, 
                         default_parameter_values = {'energy':0.5*u.TeV}, 
                         *args, **kwargs)
        





    

# Just a little helper class
class FermiGaggeroDiffusePrior(TwoCompFluxPrior):
    """
    A subclass of SourceFluxDiscreteLogPrior for modeling the Fermi-Gaggero diffuse astrophysical background prior.

    Args:
        energy_axis (np.ndarray): The energy axis for the prior (TeV).
        longitudeaxis (np.ndarray): The longitude axis for the prior (degrees).
        latitudeaxis (np.ndarray): The latitude axis for the prior (degrees).
        irf (IRF_LogLikelihood): Instrument Response Function log likelihood instance, providing `log_aeff`.
        normalise (bool, optional): Whether to normalize the prior. Defaults to True.
        logspace_integrator (callable, optional): Integrator function for normalization in log space. Defaults to logspace_riemann.

    This class constructs a prior based on the Fermi-Gaggero diffuse background model, incorporating instrument
    response functions and allowing for normalization over the specified energy, longitude, and latitude axes.
    """

    def __init__(self, 
                 axes: list[np.ndarray[u.Quantity]]| tuple[np.ndarray[u.Quantity]]=None,
                 binning_geometry: GammaBinning = None,
                 *args, **kwargs):
        """Initializes an instance of FermiGaggeroDiffusePrior, a subclass of DiscreteLogPrior, 
        designed for modeling the Fermi-Gaggero diffuse astrophysical background prior with respect 
        to given energy, longitude, and latitude axes.

        Args:
            energy_axis (np.ndarray): The energy axis for which the prior is defined, typically in GeV.
            longitudeaxis (np.ndarray): The longitude axis over which the prior is defined, in degrees.
            latitudeaxis (np.ndarray): The latitude axis over which the prior is defined, in degrees.
            irf (IRF_LogLikelihood): An instance of IRF_LogLikelihood, providing the instrument response function, 
                                     specifically `log_aeff`, the log of the effective area.
            normalise (bool, optional): If True, the log prior will be normalized over the specified axes. 
                                        Defaults to True.
            logspace_integrator (callable, optional): A function used for normalization of the log prior in log space. 
                                                      Defaults to logspace_riemann.

        This class utilizes the Fermi-Gaggero model for the diffuse astrophysical gamma-ray background, 
        incorporating the effects of the instrument's effective area as described by the provided IRF.
        """

        self._create_geometry(axes=axes, binning_geometry=binning_geometry)

        super().__init__(
            name='FermiGaggeroDiffusePrior',
            spectral_class = PowerLaw, 
            spatial_class = FermiIEM_Morphology, 
            axes=self.binning_geometry.axes,
            binning_geometry=self.binning_geometry,
            spectral_class_kwds = {'default_parameter_values': {'index':2.41, 'phi0':1.36*1e-8}}, #*u.Unit("TeV-1 cm-2 s-1 sr-1")},
            *args, **kwargs
        )
        

        self.unit = fermi_obs_rate_units

    

