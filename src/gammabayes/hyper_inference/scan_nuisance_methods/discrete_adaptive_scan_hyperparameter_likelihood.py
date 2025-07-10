from gammabayes.hyper_inference.scan_nuisance_methods.discrete_brute_scan_hyperparameter_likelihood import DiscreteBruteScan

from gammabayes.utils import (
    iterate_logspace_integration, 
    logspace_riemann, 
)


from gammabayes.utils.event_axes import derive_edisp_bounds, derive_psf_bounds
from gammabayes.utils.config_utils import save_config_file
from gammabayes import GammaObs, update_with_defaults, bound_axis

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os, warnings, logging, time, functools, numpy as np, h5py
from astropy import units as u
from icecream import ic

class DiscreteAdaptiveScan(DiscreteBruteScan):
    """
    A class that extends DiscreteBruteScan to implement adaptive bounding for nuisance parameters.

    Args:
        log_priors (list[DiscreteLogPrior] | tuple[DiscreteLogPrior], optional): 
            Priors for the log probabilities of discrete input values. Defaults to None.
        
        log_likelihood (callable, optional): A callable object to compute 
            the log likelihood. Defaults to None.
        
        axes (list[np.ndarray] | tuple[np.ndarray] | None, optional): Axes 
            that the measured event data can take. Defaults to None.
        
        nuisance_axes (list[np.ndarray] | tuple[np.ndarray] | None, optional): 
            Axes that the nuisance parameters can take. Defaults to None.
        
        parameter_specifications (dict, optional): Specifications for 
            parameters involved in the likelihood estimation. Defaults to an empty dictionary.
        
        log_likelihoodnormalisation (np.ndarray | float, optional): 
            Normalization for the log likelihood. Defaults to 0.
        
        log_margresults (np.ndarray | None, optional): Results of 
            marginalization, expressed in log form. Defaults to None.
        
        mixture_param_specifications (list[np.ndarray] | tuple[np.ndarray] | None, optional): 
            Specifications for the mixture fraction parameters. Defaults to None.
        
        log_hyperparameter_likelihood (np.ndarray | float, optional): 
            Log likelihood of hyperparameters (penultimate result). Defaults to 0.
        
        log_posterior (np.ndarray | float, optional): Log of the 
            hyperparameter posterior probability (final result of class). 
            Defaults to 0.
        
        iterative_logspace_integrator (callable, optional): A callable 
            function for iterative integration in log space. 
            Defaults to iterate_logspace_integration.
        
        logspace_integrator (callable, optional): A callable function for 
            performing single dimension integration in log space. 
            Defaults to logspace_riemann.
        
        log_prior_matrix_list (list[np.ndarray] | tuple[np.ndarray], optional): 
            List or tuple of matrices representing the log of the prior matrices
            for the given hyperparameters over the nuisance parameter axes. 
            Defaults to None.
        
        log_marginalisation_regularisation (float, optional): Regularisation 
            parameter for log marginalisation, should not change normalisation
            of final result but will minimise effects of numerical instability. Defaults to None.

        bounds (list[ str, float], optional): Radii for bounding of nuisance 
            parameter axes. Defaults to None.

        bounding_percentiles (list[float], optional): If bounds is not given then
            these values, the sigmas the given likelihood are used to produce the 
            bounds. This parameter indicates the number of samples over the 
            measured value parameter space are included by the specified 
            number of sigmas
            (e.g. if 4 sigma and 90 are given 90% of the distributions restricted 
            by the bounds contain their respective 4 sigma contours)
            Defaults to [90 ,90].

        bounding_sigmas (list[int], optional):  If bounds is not given then
            these the sigmas, the percentiles and the given likelihood are 
            used to produce the bounds. This parameter indicates the number 
            of sigma values to be contained. 
            (e.g. if 4 sigma and 90 are given 90% of the distributions restricted 
            by the bounds contain their respective 4 sigma contours)
            Defaults to [4,4].
    """




    def __init__(self, *args, 
                 bounds: list[ str, u.Quantity] = None,
                 bounding_percentiles: list[float]      = [95 ,95],
                 bounding_sigmas: list[int]     = [5,5],
                 **kwargs):
        """
        Initializes a discrete adaptive scan hyperparameter likelihood object.

        Args:
            log_priors (list[DiscreteLogPrior] | tuple[DiscreteLogPrior], optional): 
                Priors for the log probabilities of discrete input values. Defaults to None.
            
            log_likelihood (callable, optional): A callable object to compute 
                the log likelihood. Defaults to None.
            
            axes (list[np.ndarray] | tuple[np.ndarray] | None, optional): Axes 
                that the measured event data can take. Defaults to None.
            
            nuisance_axes (list[np.ndarray] | tuple[np.ndarray] | None, optional): 
                Axes that the nuisance parameters can take. Defaults to None.
            
            parameter_specifications (dict, optional): Specifications for 
                parameters involved in the likelihood estimation. Defaults to an empty dictionary.
            
            log_likelihoodnormalisation (np.ndarray | float, optional): 
                Normalization for the log likelihood. Defaults to 0.
            
            log_margresults (np.ndarray | None, optional): Results of 
                marginalization, expressed in log form. Defaults to None.
            
            mixture_param_specifications (list[np.ndarray] | tuple[np.ndarray] | None, optional): 
                Specifications for the mixture fraction parameters. Defaults to None.
            
            log_hyperparameter_likelihood (np.ndarray | float, optional): 
                Log likelihood of hyperparameters (penultimate result). Defaults to 0.
            
            log_posterior (np.ndarray | float, optional): Log of the 
                hyperparameter posterior probability (final result of class). 
                Defaults to 0.
            
            iterative_logspace_integrator (callable, optional): A callable 
                function for iterative integration in log space. 
                Defaults to iterate_logspace_integration.
            
            logspace_integrator (callable, optional): A callable function for 
                performing single dimension integration in log space. 
                Defaults to logspace_riemann.
            
            log_prior_matrix_list (list[np.ndarray] | tuple[np.ndarray], optional): 
                List or tuple of matrices representing the log of the prior matrices
                for the given hyperparameters over the nuisance parameter axes. 
                Defaults to None.
            
            log_marginalisation_regularisation (float, optional): Regularisation 
                parameter for log marginalisation, should not change normalisation
                of final result but will minimise effects of numerical instability. Defaults to None.

            bounds (list[ str, float], optional): Radii for bounding of nuisance 
                parameter axes. Defaults to None.

            bounding_percentiles (list[float], optional): If bounds is not given then
                these values, the sigmas the given likelihood are used to produce the 
                bounds. This parameter indicates the number of samples over the 
                measured value parameter space are included by the specified 
                number of sigmas
                (e.g. if 4 sigma and 90 are given 90% of the distributions restricted 
                by the bounds contain their respective 4 sigma contours)
                Defaults to [90 ,90].

            bounding_sigmas (list[int], optional):  If bounds is not given then
                these the sigmas, the percentiles and the given likelihood are 
                used to produce the bounds. This parameter indicates the number 
                of sigma values to be contained. 
                (e.g. if 4 sigma and 90 are given 90% of the distributions restricted 
                by the bounds contain their respective 4 sigma contours)
                Defaults to [4,4].
        """
      
        # Initialisation of the base class discrete_brute_scan_hyperparameter_likelihood
        super().__init__(*args, **kwargs)

        self.bounds = bounds

        # If the bounds aren't given then they are calculated based on the input log likelihood
        if self.bounds is None:
            _, logeval_bound               = derive_edisp_bounds(irf_loglike=self.log_likelihood, percentile=bounding_percentiles[0], sigmalevel=bounding_sigmas[0])
            lonlact_bound                   = derive_psf_bounds(irf_loglike=self.log_likelihood, percentile=bounding_percentiles[1], sigmalevel=bounding_sigmas[1], 
                                                                axis_buffer=1, parameter_buffer = np.squeeze(np.ptp(self.nuisance_axes[1]))/2)
            self.bounds = [['log10', logeval_bound], ['linear', lonlact_bound], ['linear', lonlact_bound]]

        logging.info(f"Bounds: {self.bounds}")
    

    def observation_nuisance_marg(self, 
                                  event_vals: np.ndarray | GammaObs, 
                                  log_prior_matrix_list: list[np.ndarray],
                                  loglike_kwargs=None):
        """
        Calculates the marginal log likelihoods for observations by integrating over nuisance parameters with 
        additional bound handling specific to this class.

        Args:
            event_vals (np.ndarray | GammaObs): The event values for which the marginal log likelihoods are to be 
                                                computed. Can be either a numpy ndarray or an GammaObs object.
            log_prior_matrix_list (list[np.ndarray]): A list of numpy ndarrays representing log prior matrices.

        Returns:
            np.ndarray: An array of marginal log likelihood values corresponding to each set of event values 
                        and each log prior matrix.

        Notes:
            - This method first adjusts the nuisance parameter axes explored based on the bounds specified 
              in the class (using the `bound_axis` method).
            - It creates a meshgrid of event values and restricted axes, considering the bounds.
            - Computes log likelihood values, adjusting them with their respective normalizations considering the indices 
              from the bounds.
            - The method then integrates these values with the log prior matrices, accounting for the rearrangement 
              of axes due to bounds.
        """

        if loglike_kwargs is None:
            loglike_kwargs = {}

        temp_axes_and_indices = [
            bound_axis(
                axis, 
                bound_type=bound_info[0], 
                bound_radii=bound_info[1], 
                estimated_val=event_val
                ) for bound_info, axis, event_val in zip(self.bounds, self.log_likelihood.dependent_axes, event_vals)]

        temp_axes = [temp_axis_info[0] for temp_axis_info in temp_axes_and_indices]

        unit_list = []
        # Presuming that the units are the same for the events and the axes
        for event_val in event_vals:
            unit_list.append(event_val.unit)

        for temp_axis in temp_axes:
            unit_list.append(1.)

        num_edisp_params = len(self.edisp_parameter_axes_by_name)
        num_psf_params = len(self.psf_parameter_axes_by_name)

        meshvalues  = np.meshgrid(*event_vals, *temp_axes, *self.edisp_parameter_axes_by_name.values(), *self.psf_parameter_axes_by_name.values(), indexing='ij')

        index_meshes = np.ix_( *[temp_axis_info[1] for temp_axis_info in temp_axes_and_indices])

        flattened_meshvalues = [meshmatrix.flatten() for meshmatrix, unit in zip(meshvalues, unit_list)]

        edisp_parameter_axes_by_kwarg = {key:flattened_mesh_axis for key, flattened_mesh_axis in zip(self.edisp_parameter_axes_by_name.keys(), flattened_meshvalues[6:6+num_edisp_params])}
        psf_parameter_axes_by_kwarg = {key:flattened_mesh_axis for key, flattened_mesh_axis in zip(self.psf_parameter_axes_by_name.keys(), flattened_meshvalues[6+num_edisp_params:6+num_edisp_params+num_psf_params])}
        
        log_likelihoodvalues = np.squeeze(self.log_likelihood(*flattened_meshvalues[:6], 
                                                              edisp_parameters = edisp_parameter_axes_by_kwarg, 
                                                              psf_parameters = psf_parameter_axes_by_kwarg, **loglike_kwargs).reshape(meshvalues[0].shape))

        log_likelihoodvalues = log_likelihoodvalues - self.log_likelihoodnormalisation[*index_meshes, ...]

        try:
            for idx, axis in enumerate(temp_axes):
                temp_axes[idx] = axis.value
        except:
            pass

        all_log_marg_results = []

        for log_prior_matrices in log_prior_matrix_list:
            # Transpose is because of the convention for where I place the 
                # axes of the true values (i.e. first three indices of the prior matrices)
                # and numpy doesn't support this kind of matrix addition nicely

            logintegrandvalues = (
                np.squeeze(log_prior_matrices).T[
                    ...,
                    *[np.newaxis]*num_psf_params,
                    *[np.newaxis]*num_edisp_params,
                    index_meshes[2].T,
                    index_meshes[1].T,
                    index_meshes[0].T
                ] + np.squeeze(log_likelihoodvalues).T).T
            
            single_parameter_log_margvals = iterate_logspace_integration(logintegrandvalues,   
                axes=temp_axes, axisindices=[0,1,2])
            
            if num_edisp_params>0:
                single_parameter_log_margvals = iterate_logspace_integration((single_parameter_log_margvals.T+self.edisp_parameter_prior_logvalue_mesh.T).T,   
                axes=list(self.edisp_parameter_axes_by_name.values()), axisindices=list(range(num_edisp_params)))


            if num_psf_params>0:
                single_parameter_log_margvals = iterate_logspace_integration( (single_parameter_log_margvals.T+self.psf_parameter_prior_logvalue_mesh.T).T,   
                axes=list(self.psf_parameter_axes_by_name.values()), axisindices=list(range(num_psf_params)))

            all_log_marg_results.append(np.squeeze(single_parameter_log_margvals))

        
        return np.array(all_log_marg_results, dtype=object)


    def pack_data(self, reduce_mem_consumption: bool = True, h5f=None, file_name='temporary_file.h5'):
        """
        Packs the information contained within this class instance into a dictionary, including the additional 
        bounds information specific to this class. This method extends the `_pack_data` method from the base class 
        by adding the bounds used for restricted parameter range marginalisation.

        Args:
            reduce_mem_consumption (bool, optional): A boolean parameter indicating whether to omit the input 
                                                    prior and likelihoods in order to save on memory consumption. 
                                                    Defaults to True, which means these large objects will not be saved.
            h5f (h5py.File, optional): An open h5py file object to save the data into. Defaults to None, in which 
                                       case a new file is created with the specified file_name.
            file_name (str, optional): The name of the file to save the data into. Defaults to 'temporary_file.h5'.

        Returns:
            h5py.File: An h5py file object containing the packed data of the class instance.

        Notes:
            - The base class method `_pack_data` is utilized to save common attributes into the dictionary.
            - This method specifically adds the `bounds` attribute, which is unique to this class.
            - The `reduce_mem_consumption` parameter is passed down to the base class method to control whether 
              large objects like input priors and likelihoods are included in the packed data.
        """
        if h5f is None:
            h5f = h5py.File(file_name, 'w-')  # Replace 'temporary_file.h5' with desired file name
        # _pack_data method in the brute force class works kind of as
            # as a base method saving the important values of the class
            # into a dictionary. We just need to add the level of restriction
            # used for the marginalisation
        
        
        h5f = self._pack_data(reduce_mem_consumption=reduce_mem_consumption, h5f=h5f)

        # Only 'new' information in this daughter class are 
            # the bounds used for the restricted parameter range
            # marginalisation
        
        bound_types = [bound[0] for bound in self.bounds]
        bound_values = [bound[1] for bound in self.bounds]

        dt = h5py.string_dtype(encoding='utf-8', length=max(len(s) for s in bound_types))

        string_ds = h5f.create_dataset('bound_types', (len(bound_types),), dtype=dt)
        string_ds[:] = bound_types

        h5f.create_dataset('bound_values', data=bound_values)

        return h5f
    

    
    @classmethod
    def load(cls, file_name:str, overriding_class_input_dict:dict = {}, *args, **kwargs):
        """
        Loads a DiscreteAdaptiveScan instance from a file.

        Args:
            file_name (str): The name of the file to load the data from.
            overriding_class_input_dict (dict, optional): A dictionary to override the class input parameters.
                                                         Defaults to an empty dictionary.

        Returns:
            DiscreteAdaptiveScan: An instance of the DiscreteAdaptiveScan class with the loaded data.
        """
        
        class_input_dict = cls.unpack(file_name, overriding_class_input_dict=overriding_class_input_dict)

        logging.info("Unpacking of stem class parameters successful")
        
        with h5py.File(file_name, 'r') as h5f:

            bound_types = np.array(h5f['bound_types'])
            bound_values = np.array(h5f['bound_values'])

            bound_types_decoded = [s.decode('utf-8') if isinstance(s, bytes) else s for s in bound_types]
            
            class_input_dict['bounds'] = [[bound_type, bound_value] for bound_type, bound_value in zip(bound_types_decoded, bound_values)]

        logging.info("Unpacking of bounds parameter successful")

        return cls(no_priors_on_init=True, *args, **class_input_dict, **kwargs)
            

