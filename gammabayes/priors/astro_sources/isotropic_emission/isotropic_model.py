import numpy as np
from astropy import units as u

from gammabayes.priors.core import TwoCompFluxPrior
from gammabayes import GammaBinning

from gammabayes.priors.spectral_components import BrokenPowerLaw, PowerLaw
from gammabayes.priors.spatial_components import IsotropicSpatial_PriorComp




class IsotropicBrokenPowerLaw(TwoCompFluxPrior):

    def __init__(self, 
                 axes: list[np.ndarray[u.Quantity]]| tuple[np.ndarray[u.Quantity]]=None,
                 binning_geometry: GammaBinning = None,
                 *args, **kwargs):
        
        name = kwargs.get("name")

        if name is None:
            name='IGRB'
        
        kwargs['name'] = name

        spectral_class_kwds = kwargs.get("spectral_class_kwds")

        __default_spectral_class_kwds = {'default_parameter_values': {'index':2.3, 'cutoff_energy_TeV':0.3, 'phi0':1e-1}} #*u.Unit("TeV-1 cm-2 s-1 sr-1")}}
        
        if spectral_class_kwds is None:
            spectral_class_kwds = __default_spectral_class_kwds

        if 'default_parameter_values' in spectral_class_kwds.keys():
            for key, item in __default_spectral_class_kwds['default_parameter_values'].items():
                spectral_class_kwds['default_parameter_values'].setdefault(key, item)

        kwargs['spectral_class_kwds'] = spectral_class_kwds

        
        self._create_geometry(axes=axes, binning_geometry=binning_geometry)

        super().__init__(
            spectral_class = BrokenPowerLaw, 
            spatial_class = IsotropicSpatial_PriorComp, 
            axes=self.binning_geometry.axes,
            binning_geometry=self.binning_geometry,
            *args, **kwargs
        )
        

class IsotropicPowerLaw(TwoCompFluxPrior):

    def __init__(self, 
                 axes: list[np.ndarray[u.Quantity]]| tuple[np.ndarray[u.Quantity]]=None,
                 binning_geometry: GammaBinning = None,
                 *args, **kwargs):
        
        name = kwargs.get("name")

        if name is None:
            name='IsotropicGRB'
        
        kwargs['name'] = name

        spectral_class_kwds = kwargs.get("spectral_class_kwds")

        __default_spectral_class_kwds = {'default_parameter_values': {'index':2.5, 'phi0':1e-1}}#*u.Unit("TeV-1 cm-2 s-1 sr-1")}}
        if spectral_class_kwds is None:
            spectral_class_kwds = __default_spectral_class_kwds

        if 'default_parameter_values' in spectral_class_kwds.keys():
            for key, item in __default_spectral_class_kwds['default_parameter_values'].items():
                spectral_class_kwds['default_parameter_values'].setdefault(key, item)
        else:
            spectral_class_kwds['default_parameter_values'] = __default_spectral_class_kwds['default_parameter_values']

        kwargs['spectral_class_kwds'] = spectral_class_kwds

        
        self._create_geometry(axes=axes, binning_geometry=binning_geometry)



        super().__init__(
            spectral_class = PowerLaw, 
            spatial_class = IsotropicSpatial_PriorComp, 
            axes=self.binning_geometry.axes,
            binning_geometry=self.binning_geometry,
            *args, **kwargs
        )