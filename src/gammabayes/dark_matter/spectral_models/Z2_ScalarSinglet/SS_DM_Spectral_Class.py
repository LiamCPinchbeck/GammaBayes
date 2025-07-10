import numpy as np, copy
from astropy import units as u
from os import path
ScalarSinglet_Folder_Path = path.dirname(__file__)

from gammabayes.dark_matter.spectral_models.core import DM_ContinuousEmission_Spectrum, CSVDictionary
import time

class Z2_ScalarSinglet(DM_ContinuousEmission_Spectrum):
    """
    Implements the Z2 Scalar Singlet dark matter model for calculating the gamma-ray emission spectrum 
    from dark matter annihilation. This specialized subclass of `DM_ContinuousEmission_Spectrum` leverages 
    annihilation fractions and parameter axes derived from partial annihilation cross sections from two
    user changeable sources: 'darkSUSY' or 'dimauromattia'. Referring to the results from the paper with
    ArXiV ID 2305.11937 by Di Mauro+, with results available on dimauromattia's GitHub page. For 
    Di Mauro+ results, values above approximately 20 TeV are taken to be the same as at 20 TeV. This 
    approximation is done because the dominating channels (W, Z and kind of Higgs which makes up 5-8% of the
    total) are approximately constant at this point
    anyway.

    Utilizing these model-specific parameters, it enables precise spectral predictions crucial for dark matter 
    indirect detection studies. The class reads annihilation fractions from a CSV file, organizing these into 
    a structured format for spectral computation.

    Initialization:
    - Mass and lambda (self-interaction strength) parameters are extracted from the CSV file and adjusted from GeV to TeV.
    - Annihilation fractions are formatted into a dictionary, setting up the model for spectral calculations.

    Attributes:
        interp_file (str): Name of the interpolation file without extension, used to differentiate between 
                           various model parameterizations.

    Args:
        interp_file (str, optional): Specifies the interpolation file to use for model parameters. 
                                     Defaults to 'darkSUSY'.
        *args: Additional positional arguments passed to the parent class constructor.
        **kwargs: Additional keyword arguments passed to the parent class constructor, including options for 
                  default parameter values and other configurations.
    """


    def __init__(self, interp_file = 'darkSUSY', *args, **kwargs):
        """
        Initializes the Z2_ScalarSinglet class with data specific to the model, preparing it for emission spectrum calculations.

        Args:
            interp_file (str, optional): File name (without extension) of the interpolation data within the 
                                         `annihilation_ratio_data` directory. This file contains the model-specific 
                                         annihilation fractions and parameter axes. Defaults to 'darkSUSY'.
            *args: Variable length argument list for the parent class's constructor.
            **kwargs: Arbitrary keyword arguments for the parent class's constructor. This includes options to specify
                      default parameter values such as mass and self-interaction strength (`lahS`).
        """

        annihilation_ratio_data_dict = CSVDictionary(ScalarSinglet_Folder_Path+f'/annihilation_ratio_data/annihilation_ratios__{interp_file}.csv', delimiter=' ')

        # Extracting the annihilation ratios for the Scalar Singlet model
        mass_axis, lahS_axis = np.unique(annihilation_ratio_data_dict['mS [GeV]'])/1e3, np.unique(annihilation_ratio_data_dict['lahS'])

        self._SS_ratios_dict = copy.deepcopy(annihilation_ratio_data_dict)

        del self._SS_ratios_dict['mS [GeV]']
        del self._SS_ratios_dict['lahS']
        
        parameter_interpolation_values = [mass_axis, lahS_axis]
        parameter_axes_shapes = (lahS_axis.size, mass_axis.size)

        for channel in self._SS_ratios_dict.keys():
            self._SS_ratios_dict[channel] = self._SS_ratios_dict[channel].reshape(parameter_axes_shapes).T


        super().__init__(annihilation_fractions = self._SS_ratios_dict, 
                         parameter_interpolation_values = parameter_interpolation_values,
                         default_parameter_values={'mass':1.0, 'lahS':0.1},
                         *args, **kwargs)
        





