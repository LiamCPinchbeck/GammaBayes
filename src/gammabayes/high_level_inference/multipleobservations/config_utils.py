import os, sys, yaml, astropy.units as u, matplotlib.pyplot as plt, numpy as np
from gammabayes import GammaBinning, GammaLogExposure, GammaObs, Parameter, ParameterSet, ParameterSetCollection
from gammabayes.utils.config_utils import (
    create_true_axes_from_config, 
    create_recon_axes_from_config, 
)
import copy

def from_config_file(cls, config_file_name, skip_irf_norm=True):
    with open(config_file_name, 'r') as file:
        config_dict = yaml.safe_load(file)

    return cls.from_config_dict(config_dict, skip_irf_norm=skip_irf_norm)

def from_config_dict(cls, config_dict, skip_irf_norm=True):
    config_dict = copy.deepcopy(config_dict)
    true_axes       = create_true_axes_from_config(config_dict)
    recon_axes      = create_recon_axes_from_config(config_dict)

    true_binning_geometry = GammaBinning(*true_axes)
    recon_binning_geometry = GammaBinning(*recon_axes)


    pointing_dirs = [np.array(direction)*u.deg for direction in config_dict['pointing_directions']]

    observation_times = [u.Quantity(_obs_time) for _obs_time in config_dict['observation_times']]

    mixture_layout = config_dict.get("mixture_layout")
    observational_prior_parameter_specifications = config_dict.get("observational_prior_parameter_specifications")
    shared_parameter_specifications = config_dict.get("shared_observational_prior_parameter_specifications")
    mixture_parameter_specifications = config_dict.get("mixture_parameter_specifications")
    true_mixture_specifications = config_dict.get("true_mixture_specifications")
    save_path = config_dict.get("save_path", "")
    observational_prior_models  = config_dict.get("observational_prior_models")

    dark_matter_model_specifications = config_dict.get("dark_matter_model_specifications")

    marginalisation_bounds = config_dict.get("marginalisation_bounds", [['log10', 1.0*u.TeV], ['linear', 1.0*u.deg], ['linear', 1.0*u.deg]])
    marginalisation_method = config_dict.get("marginalisation_method",  "DiscreteAdaptiveScan")


    irf_kwargs = construct_loglike_normalisations(irf_specifications=config_dict.get("irf_specifications"), skip_irf_norm=skip_irf_norm)


    return cls(
        true_binning_geometry=true_binning_geometry,
        recon_binning_geometry=recon_binning_geometry,
        pointing_dirs=pointing_dirs,
        observation_times=observation_times,

        save_path=save_path,

        prior_parameter_specifications=observational_prior_parameter_specifications,
        shared_parameter_specifications=shared_parameter_specifications,

        mixture_layout= mixture_layout,
        mixture_parameter_specifications=mixture_parameter_specifications,
        dark_matter_model_specifications=dark_matter_model_specifications,
        observational_prior_models=observational_prior_models,
        true_mixture_specifications=true_mixture_specifications,

        marginalisation_bounds=marginalisation_bounds,
        marginalisation_method=marginalisation_method,

        config_dict=config_dict,
        
        **irf_kwargs

    )


def construct_loglike_normalisations(irf_specifications:dict, skip_irf_norm=True):

    log_irf_norm_matrix_path = ""
    log_psf_norm_matrix_path = ""
    log_edisp_norm_matrix_path = ""
    return {"irf_log_norm_matrix":   None, 
            "edisp_log_norm_matrix": None, "psf_log_norm_matrix": None, "skip_irf_norm_setup": skip_irf_norm, 
            "log_irf_norm_matrix_path":    log_irf_norm_matrix_path, 
            "log_psf_norm_matrix_path":    log_psf_norm_matrix_path, 
            "log_edisp_norm_matrix_path":  log_edisp_norm_matrix_path}
















