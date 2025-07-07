from .parameter_set_class import ParameterSet
import logging, warnings, numpy as np, yaml, sys, os, torch
from scipy import integrate, special, interpolate, stats
import random, time, pickle
from tqdm import tqdm
from scipy.stats import norm as norm1d
from scipy.interpolate import RegularGridInterpolator
from astropy import units as u
from astropy.units import Quantity
from os import path
resources_dir = path.join(path.dirname(__file__), '../package_data')


def update_with_defaults(target_dict, default_dict):
    """
    Updates the target dictionary in place, adding missing keys from the default dictionary.

    Args:
        target_dict (dict): The dictionary to be updated.
        default_dict (dict): The dictionary containing default values.
    """
    for key, value in default_dict.items():
        target_dict.setdefault(key, value)


def haversine(lon1, lat1, lon2, lat2):
        # Convert degrees to radians
    lon1_rad = torch.deg2rad(lon1)
    lat1_rad = torch.deg2rad(lat1)
    lon2_rad = torch.deg2rad(lon2)
    lat2_rad = torch.deg2rad(lat2)

    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula components
    a = torch.sin(dlat / 2.0)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2.0)**2
    
    # Clamp 'a' to a valid range [0, 1] to prevent numerical issues with sqrt for very small
    # negative values that might occur due to floating point inaccuracies.
    a = torch.clamp(a, 0.0, 1.0)

    # Angular distance 'c' (central angle in radians)
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(1.0 - a))

    return torch.rad2deg(c)

def power_law(energy: float|Quantity, index: float, phi0: int|Quantity =1) -> float|Quantity:
    """
    Evaluates a power law function.

    Args:
        energy (float | Quantity): Energy values.
        index (float): Power law index.
        phi0 (int | Quantity, optional): Normalization constant. Defaults to 1.

    Returns:
        float | Quantity: Computed power law values.
    """
    warnings.warn("power_law will be deprecated after version 0.1.16. Please use the provided function in the prior.spectral components module.")
    return phi0*energy**(index)


def _handle_parameter_specification(
        parameter_specifications: dict | ParameterSet,
        num_required_sets: int = None,
        _no_required_num=False):
    """
    Processes and validates parameter specifications against the target priors.

    Parameters:
        parameter_specifications (dict | ParameterSet): Parameter specifications.
        
        log_target_priors (list[ParameterSet] | dict[dict], optional): Target priors.

    Raises:
        ValueError: If the number of hyperparameter axes specified exceeds the number of priors unless
        'self.no_priors_on_init' is True.
    """
    _num_parameter_specifications = len(parameter_specifications)
    formatted_parameter_specifications = []*_num_parameter_specifications

    if _num_parameter_specifications>0:

        if type(parameter_specifications)==dict:

            for single_prior_parameter_specifications in parameter_specifications.items():

                parameter_set = ParameterSet(single_prior_parameter_specifications)

                formatted_parameter_specifications.append(parameter_set)

        elif type(parameter_specifications)==list:
            formatted_parameter_specifications = [
                ParameterSet(
                    single_prior_parameter_specification
                    ) for single_prior_parameter_specification in parameter_specifications
                ]

    if num_required_sets is not None:
        _num_priors = num_required_sets
    else:
        _num_priors = _num_parameter_specifications

    if not _no_required_num or (num_required_sets is not None):

        diff_in_num_hyperaxes_vs_priors = _num_priors-_num_parameter_specifications

        if diff_in_num_hyperaxes_vs_priors<0:
            raise ValueError(f'''
You have specifed {np.abs(diff_in_num_hyperaxes_vs_priors)} more hyperparameter axes than priors.''')
        
        elif diff_in_num_hyperaxes_vs_priors>0:
            warnings.warn(f"""
You have specifed {diff_in_num_hyperaxes_vs_priors} less hyperparameter axes than priors. 
Assigning empty hyperparameter axes for remaining priors.""")
            
            _num_parameter_specifications = len(formatted_parameter_specifications)
            
            for __idx in range(_num_parameter_specifications, _num_priors):
                formatted_parameter_specifications.append(ParameterSet())


    return formatted_parameter_specifications


def _handle_nuisance_axes(nuisance_axes: list[np.ndarray],
                            log_likelihood=None, log_prior=None):
    """
    Handles the assignment and retrieval of nuisance axes. 
    This method first checks if `nuisance_axes` is provided. If not, it attempts to retrieve nuisance axes 
    from `log_likelihood` or `log_prior`. If neither is available, it raises an exception.

    Args:
        nuisance_axes (list[np.ndarray]): A list of numpy arrays representing the nuisance axes.

    Raises:
        Exception: Raised if `nuisance_axes` is not provided and cannot be retrieved from either 
                `log_likelihood` or `log_proposal_prior` OR `log_target_priors`.

    Returns:
        list[np.ndarray]: The list of numpy arrays representing the nuisance axes. This can be either the 
                        provided `nuisance_axes`, or retrieved from `log_likelihood` or `log_priors`.
    """
    if nuisance_axes is None:
        try:
            return log_likelihood.nuisance_axes
        except AttributeError:
            try:
                return log_prior.axes
            except AttributeError:
                raise Exception("Dependent value axes used for calculations not given.")
                
    return nuisance_axes






def save_to_pickle(filename, object_to_save, write_mode='wb'):
    """
    Saves an object to a file using pickle with a specified write mode.

    Args:
        filename (str): The name of the file to save the object.
        object_to_save (object): The object to be saved.
        write_mode (str, optional): The mode in which the file is opened. Defaults to 'wb'.
    """
    with open(filename, write_mode) as file:
        pickle.dump(object_to_save, file)


def load_pickle(filename, load_mode='rb'):
    """
    Loads an object from a pickle file with a specified load mode.

    Args:
        filename (str): The name of the file to load the object from.
        load_mode (str, optional): The mode in which the file is opened. Defaults to 'rb'.

    Returns:
        object: The loaded object.
    """
    with open(filename, load_mode) as file:
        loaded_object = pickle.load(file)

    return loaded_object








