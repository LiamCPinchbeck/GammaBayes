import numpy as np, copy
from os import path
Z5_Folder_Path = path.dirname(__file__)

from gammabayes.dark_matter.spectral_models.core import multi_comp_dm_spectrum
import time


result_dict = {}

import h5py, numpy as np, matplotlib.pyplot as plt



class extract_Z5_h5_results():
    """
    A class to extract Z5 dark matter results from HDF5 files.

    Methods:
        extract(file_name): Extracts the annihilation ratios and parameter values from the HDF5 file.
    """

    @classmethod
    def extract(self, file_name):
        """
        Extracts the annihilation ratios and parameter values from the HDF5 file.

        Args:
            file_name (str): The path to the HDF5 file.

        Returns:
            tuple: A tuple containing:
                - result_dict (dict): A nested dictionary with initial and final state keys and their corresponding ratios.
                - list: A list containing the unique mass values for mdm1 and mdm2.
        """
        with h5py.File(file_name, 'r') as h5f:
            initial_state_options   = list(h5f['initial_state_options'])
            initial_state_options   = [keystr.decode('ascii') for keystr in initial_state_options]

            inum = 2
            fnum = 0
            conv_keys               = list(h5f['final_state_options'])

            conv_keys               = [keystr.decode('ascii') for keystr in conv_keys]
            mdm1 = np.unique(h5f['mdm1'])
            mdm2 = np.unique(h5f['mdm2'])
            total_ratios = 0

            for init_key in initial_state_options:
                for f_key in conv_keys:
                    total_ratios = total_ratios + np.asarray(h5f[init_key][f_key])

            result_dict = {}
            for init_key in initial_state_options:
                result_dict[init_key] = {}
                for f_key in conv_keys:
                    result_dict[init_key][f_key] =  np.asarray(h5f[init_key][f_key]).reshape(mdm1.size, mdm2.size)
            
        self.result_dict = result_dict

        return result_dict, [mdm1, mdm2]



class Z5_DM_spectra(multi_comp_dm_spectrum):
    """
    A class to handle Z5 dark matter spectra using multi-component dark matter spectrum.

    Inherits from:
        multi_comp_dm_spectrum

    Methods:
        __init__(self, annihilation_ratios_nested_dict=None, parameter_interpolation_values=None, **kwargs)
    """
    def __init__(self, annihilation_ratios_nested_dict=None, parameter_interpolation_values=None, **kwargs):
        """
        Initializes the Z5_DM_spectra object with annihilation ratios and parameter interpolation values.

        Args:
            annihilation_ratios_nested_dict (dict, optional): Nested dictionary containing annihilation ratios. Defaults to None.
            parameter_interpolation_values (list, optional): List of parameter interpolation values. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """

        if annihilation_ratios_nested_dict is None or (parameter_interpolation_values is None):
            file_name = Z5_Folder_Path+'/annihilation_ratio_data/z5_101x101_ratios.h5'

            annihilation_ratios_nested_dict, parameter_interpolation_values = extract_Z5_h5_results.extract(file_name=file_name)
            parameter_interpolation_values = [param_interpolation_val/1000 for param_interpolation_val in parameter_interpolation_values]


        super().__init__(annihilation_ratios_nested_dict=annihilation_ratios_nested_dict, 
                         parameter_interpolation_values=parameter_interpolation_values, **kwargs)

    