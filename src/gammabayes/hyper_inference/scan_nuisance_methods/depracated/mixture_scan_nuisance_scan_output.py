import numpy as np, warnings, dynesty, logging
from gammabayes import apply_dirichlet_stick_breaking_direct, update_with_defaults, ParameterSet, Parameter
from gammabayes.hyper_inference.core.utils import _handle_parameter_specification
import h5py

class ScanOutput_ScanMixtureFracPosterior(object):
    """
    This class processes the scan output for analyzing mixture fraction posteriors within a Bayesian framework.
    It calculates the posterior distributions of mixture fractions by integrating over nuisance parameters,
    using a specified mixture model. This class supports handling of log marginal results from the scan,
    application of Dirichlet stick breaking for mixture models, and calculation of discrete mixture
    log hyper-likelihoods.

    Attributes:
        log_nuisance_marg_results (np.ndarray): Log marginal results for nuisance parameters from the scan.
        log_nuisance_marg_regularisation (float): Regularization term added to the log marginal results to stabilize
                                                  the numerical computations.
        mixture_parameter_specifications (ParameterSet): Specifications for the mixture model parameters, which
                                                         includes parameter names, bounds, and discretization.
    
    Args:
        log_margresults (np.ndarray): The log marginal results from the scanning process.
        mixture_parameter_specifications (list | ParameterSet, optional): Specifications for the mixture model parameters.
                                                                          If not provided, a default set is used.
        log_nuisance_marg_regularisation (float, optional): Regularization term for log marginal results to help
                                                            stabilize computations. Defaults to 0.
        prior_parameter_specifications (None, optional): Unused. Present for API consistency with other classes.
    """

    def __init__(self, 
                 log_nuisance_marg_results, 
                 mixture_parameter_specifications: list | ParameterSet = None, 
                 log_nuisance_marg_regularisation = 0., 
                 prior_parameter_specifications = None, # Argument is for consistent input to this class and "ScanOutput_StochasticMixtureFracPosterior"
                 log_hyperparameter_likelihood = np.array([0.]),
                 ):
        """
        Initializes the ScanOutput_ScanMixtureFracPosterior object with scan results and model specifications.

        Attributes:
            log_nuisance_marg_results (np.ndarray): Log marginal results for nuisance parameters from the scan.
            log_nuisance_marg_regularisation (float): Regularization term added to the log marginal results to stabilize
                                                    the numerical computations.
            mixture_parameter_specifications (ParameterSet): Specifications for the mixture model parameters, which
                                                            includes parameter names, bounds, and discretization.
        
        Args:
            log_margresults (np.ndarray): The log marginal results from the scanning process.
            mixture_parameter_specifications (list | ParameterSet, optional): Specifications for the mixture model parameters.
                                                                            If not provided, a default set is used.
            log_nuisance_marg_regularisation (float, optional): Regularization term for log marginal results to help
                                                                stabilize computations. Defaults to 0.
            prior_parameter_specifications (None, optional): Unused. Present for API consistency with other classes.
        """
        
        self.log_nuisance_marg_results = log_nuisance_marg_results
        self.log_nuisance_marg_regularisation = log_nuisance_marg_regularisation
        self.mixture_parameter_specifications = self._handle_mixture_input(mixture_parameter_specifications)
        self.log_hyperparameter_likelihood      = log_hyperparameter_likelihood



    def _handle_mixture_input(self, 
                                 mixture_param_specifications: list | dict | ParameterSet = None, 
                                 log_nuisance_marg_results: list | tuple | np.ndarray = None):
        """
        Validates and processes the input for mixture model specifications and log marginal results, ensuring they
        are in the correct format and consistent with each other. This method also handles default parameters and
        initializes the mixture parameter specifications if not provided.

        Args:
            mixture_param_specifications (list | dict | ParameterSet, optional): The specifications for the mixture
                model parameters. Can include parameter names, bounds, and discretization settings. If None, uses
                the object's existing specifications.
            log_nuisance_marg_results (list | tuple | np.ndarray, optional): Log marginal results for validating against the
                mixture parameter specifications. Not directly used here but ensures consistency.

        Returns:
            tuple: A tuple containing the processed and validated mixture_param_specifications and log_nuisance_marg_results.

        Raises:
            Exception: If the mixture parameter specifications are not provided or are inconsistent with the log
                marginal results.

        Notes:
            This method is primarily used internally to ensure the consistency and validity of input parameters
            for the mixture model.
        """

        
        if mixture_param_specifications is None:
            mixture_param_specifications = self.mixture_param_specifications
            if self.mixture_param_specifications is None:
                raise Exception("Mixture axes not specified")
        else:
            self.mixture_param_specifications = ParameterSet(mixture_param_specifications)

        if not(log_nuisance_marg_results is None):
            if len(mixture_param_specifications)>len(log_nuisance_marg_results)-1:
                raise Exception(f""""There are more mixtures than would be implied by the log marginalisation results. Please check your inputs.
Number of mixture axes is {len(mixture_param_specifications)}
and number of prior components is {len(log_nuisance_marg_results)}.""")
            elif len(mixture_param_specifications)<len(log_nuisance_marg_results)-1:
                warnings.warn("""There are less mixtures than would be implied by the log marginalisatio results.
                              Assigning remaining mixtures as linearly uniform from 0 to 1 with 21 bins.""")
                for missing_mix in range(len(mixture_param_specifications), len(log_nuisance_marg_results)-1):
                    mixture_param_specifications.append(Parameter(
                        name=f'UnknownMix{missing_mix}',
                        bounds=[0., 1.],
                        discrete=True,
                        bins=21,
                    ))
            

        return mixture_param_specifications
    
    def create_mixture_comp(self, 
                            prior_idx: int, 
                            log_margresults_for_idx:np.ndarray,
                            mix_axes_mesh:list[np.ndarray],
                            final_output_shape:list,
                            prior_axes_indices: list[list],
                            hyper_idx: int):
        """
        Constructs a single component of the mixture model for a given prior. This involves combining the log
        marginal results with mixture axis information to form the component, using Dirichlet stick breaking.

        Args:
            prior_idx (int): Index of the prior for which the component is being created.
            
            log_margresults_for_idx (np.ndarray): Log marginal results corresponding to the prior index.
            
            mix_axes_mesh (list[np.ndarray]): Meshgrid of mixture axes, providing the basis for the mixture component.
            
            final_output_shape (list): List to store the shape of the output mixture component.
            
            prior_axes_indices (list[list]): List of indices for each prior in the final output.
            
            hyper_idx (int): Index in the hyperparameter space, used for tracking the dimensionality.

        Returns:
            tuple: A tuple containing the mixture component (as a numpy array) and the updated hyperparameter index.

        Notes:
            This method applies the Dirichlet stick breaking process to create a log mixture component and updates
            tracking lists for output shape and hyperparameter indices.
        """
        # Including 'event' and mixture axes for eventual __non__ expansion into
        single_prior_axes_indices_instance = list(range(1+len(mix_axes_mesh)))

        # First index of 'log_margresults_for_idx' should just be the number of events
        for length_of_axis in log_margresults_for_idx.shape[1:]:

            final_output_shape.append(length_of_axis)

            single_prior_axes_indices_instance.append(hyper_idx)

            # Keeping track of the indices in each prior for the final output
            hyper_idx+=1

        prior_axes_indices.append(single_prior_axes_indices_instance)


        mixcomp = np.expand_dims(np.log(
            apply_dirichlet_stick_breaking_direct(mixtures_fractions=mix_axes_mesh, depth=prior_idx)), 
            axis=(*np.delete(np.arange(log_margresults_for_idx.ndim+len(mix_axes_mesh)), 
                                np.arange(len(mix_axes_mesh))+1),)) 


        mixcomp=mixcomp+np.expand_dims(log_margresults_for_idx, 
                                    axis=(*(np.arange(len(mix_axes_mesh))+1),)
                                )
        return mixcomp, hyper_idx
        
    
            
    def create_discrete_mixture_log_hyper_likelihood(self, 
                                                     batch_of_log_margresults: list | tuple | np.ndarray = None,
                                                     ) -> np.ndarray:
        """
        Calculates the log hyper-likelihood of the discrete mixture model based on batched log marginal results.
        This method adjusts the marginal results with regularization, constructs mixture components, and combines
        them to compute the overall log hyper-likelihood.

        Args:
            batch_of_log_margresults (list | tuple | np.ndarray, optional): Batched log marginal results for each
                prior. If None, uses the object's log_nuisance_marg_results.

        Returns:
            np.ndarray: The log hyper-likelihood of the hyperparameters given the batch of log marginal results.

        Notes:
            This method orchestrates the creation of mixture components for each prior and combines them to compute
            the log hyper-likelihood, taking regularization into account.
        """

        batch_of_log_margresults = [log_margresult + self.log_nuisance_marg_regularisation for log_margresult in batch_of_log_margresults]
        

        mix_axes_mesh = np.meshgrid(*self.mixture_parameter_specifications.axes, indexing='ij')

        # To get around the fact that every component of the mixture can be a different shape
        final_output_shape = [len(batch_of_log_margresults), *np.arange(len(self.mixture_parameter_specifications)), ]

        # Axes for each of the priors to __not__ expand in
        prior_axes_indices = []

        # Counter for how many hyperparameter axes we have used
        hyper_idx = len(self.mixture_parameter_specifications)+1


        # Creating components of mixture for each prior
        mixture_array_comp_list = []
        for prior_idx, log_margresults_for_idx in enumerate(batch_of_log_margresults):

            mix_comp, hyper_idx  = self.create_mixture_comp( 
                            prior_idx = prior_idx, 
                            log_margresults_for_idx = log_margresults_for_idx,
                            mix_axes_mesh=mix_axes_mesh,
                            final_output_shape=final_output_shape,
                            prior_axes_indices=prior_axes_indices,
                            hyper_idx=hyper_idx)
            
            mixture_array_comp_list.append(mix_comp)
            
        # Making all the mixture components the same shape that is compatible for adding together (in logspace)
        for _prior_idx, mixture_array in enumerate(mixture_array_comp_list):
            axis = []
            for _axis_idx in range(len(final_output_shape)):
                if not(_axis_idx in prior_axes_indices[_prior_idx]):
                    axis.append(_axis_idx)
            axis = tuple(axis)

            mixture_array   = np.expand_dims(mixture_array, axis=axis)

            mixture_array_comp_list[_prior_idx] = mixture_array

        # Now to combine the results for all the data and to 
        combined_mixture = -np.inf
        for mixture_component in mixture_array_comp_list:
            mixture_component = mixture_component - self.log_nuisance_marg_regularisation
            combined_mixture = np.logaddexp(combined_mixture, mixture_component)


        # Multiplying the likelihoods of all the events together
        log_hyperparameter_likelihood = np.sum(combined_mixture, axis=0)

        # Undoing the regularisation for proper output
        log_hyperparameter_likelihood = log_hyperparameter_likelihood

        return log_hyperparameter_likelihood
    
    def initiate_exploration(self, 
                           num_in_batches: int) -> list:
        """
        Prepares the exploration process by batching the log marginal results. This method divides the log marginal
        results into smaller batches for processing, facilitating efficient computation of the hyper-likelihood.

        Args:
            num_in_batches (int): The number of log marginal results to include in each batch.

        Returns:
            list: A list of batched log marginal results ready for exploration.
        
        Notes:
            This method is used to setup the exploration process by organizing the log marginal results into
            manageable batches.
        """
        batched_log_margresults = []
        for batch_idx in range(0, len(self.log_nuisance_marg_results[0]), num_in_batches):
            batched_log_margresults.append(
                [single_prior_log_margresults[batch_idx:batch_idx+num_in_batches, ...] for single_prior_log_margresults in self.log_nuisance_marg_results])

        self.batched_log_nuisance_marg_results =  batched_log_margresults


    def run_exploration(self) -> np.ndarray:
        """
        Executes the exploration process, computing the log hyperparameter likelihood for each batch of log
        marginal results. This method iterates over all batches, calculates their log hyper-likelihood, and
        aggregates the results.

        Returns:
            np.ndarray: The combined log hyperparameter likelihood from all batches of log marginal results.

        Notes:
            This method leverages `create_discrete_mixture_log_hyper_likelihood` to calculate the log
            hyper-likelihood for each batch and sums the log results to get the overall likelihood.
        """
        
        log_hyperparameter_likelihood_batches = [self.create_discrete_mixture_log_hyper_likelihood(
            batch_of_log_margresults=batch_of_log_margresults,
        ) for batch_of_log_margresults in self.batched_log_nuisance_marg_results]

        log_hyperparameter_likelihood = np.sum(log_hyperparameter_likelihood_batches, axis=0)

        self.log_hyperparameter_likelihood = log_hyperparameter_likelihood

        return log_hyperparameter_likelihood
    

    def _pack_data(self, h5f=None, file_name=None, save_log_hyperparameter_likelihood=False):
        """
        Private method that packs the class data into an HDF5 format.

        Equivalent to the public method for use in sub-classes.

        Args:
        h5f (h5py.File): An open HDF5 file object for writing data.
        """
        if h5f is None:
            h5f = h5py.File(file_name, 'w-')
        # Ensure mixture_parameter_specifications is properly handled if it's a ParameterSet
        if isinstance(self.mixture_parameter_specifications, ParameterSet):
            # Assuming ParameterSet has a pack method to handle its serialization
            mixture_param_group = h5f.create_group("mixture_parameter_specifications")
            self.mixture_parameter_specifications.pack(mixture_param_group)
        
        # Save log_nuisance_marg_results
        log_nuisance_marg_results_group = h5f.create_group("log_nuisance_marg_results")
        for result_idx, result in enumerate(self.log_nuisance_marg_results):
            log_nuisance_marg_results_group.create_dataset(str(result_idx), data=result)

        if save_log_hyperparameter_likelihood:
            h5f.create_dataset("log_hyperparameter_likelihood", data=self.log_hyperparameter_likelihood)
        
        # Save log_nuisance_marg_regularisation as an attribute
        h5f.attrs["log_nuisance_marg_regularisation"] = self.log_nuisance_marg_regularisation

        return h5f
    
    def pack_data(self, h5f=None, file_name=None, save_log_hyperparameter_likelihood=False):
        """
        Packs the class data into an HDF5 format (wrapper for _pack_data).

        Args:
        h5f (h5py.File): An open HDF5 file object for writing data.
        """        
        return self._pack_data(h5f=h5f, file_name=file_name, save_log_hyperparameter_likelihood=save_log_hyperparameter_likelihood)
    

    
    def save(self, file_name):
        """
        Saves the class data to an HDF5 file.

        Args:
            file_name (str): The name of the file to save the data to.
        """
        h5f = self.pack_data(file_name=file_name)
        h5f.close()
    

    @classmethod
    def load(cls, h5f=None, file_name=None):
        """
        Loads the class data from an HDF5 file.

        Args:
        file_name (str): The path to the HDF5 file to load.

        Returns:
        ScanOutput_ScanMixtureFracPosterior: An instance of the class reconstructed from the file.
        """

        if type(h5f)==str:
            file_name = h5f
            h5f = None

        if h5f is None:
            need_to_close = True
            h5f = h5py.File(file_name)


        # Load mixture_parameter_specifications
        mixture_param_group = h5f["mixture_parameter_specifications"]
        mixture_parameter_specifications = ParameterSet.load(mixture_param_group)
        
        # Load log_nuisance_marg_results
        log_nuisance_marg_results = []
        log_nuisance_marg_results_group = h5f["log_nuisance_marg_results"]
            
        # Load each dataset within the "log_marg_results" group
        # Assuming the datasets are named as "0", "1", "2", ...
        # and need to be loaded in the order they were saved
        result_indices = sorted(log_nuisance_marg_results_group.keys(), key=int)
        for result_idx in result_indices:
            result = np.asarray(log_nuisance_marg_results_group[result_idx])
            log_nuisance_marg_results.append(result)

        log_nuisance_marg_results = np.asarray(log_nuisance_marg_results, dtype=object)


        if 'log_hyperparameter_likelihood' in h5f:
            log_hyperparameter_likelihood = np.asarray(h5f["log_hyperparameter_likelihood"])
        else:
            log_hyperparameter_likelihood = np.array([0.])
        
        # Load log_nuisance_marg_regularisation
        log_nuisance_marg_regularisation = h5f.attrs["log_nuisance_marg_regularisation"]
        
        # Reconstruct the class instance
        instance = cls(log_nuisance_marg_results=log_nuisance_marg_results,
                    mixture_parameter_specifications=mixture_parameter_specifications,
                    log_nuisance_marg_regularisation=log_nuisance_marg_regularisation,
                    log_hyperparameter_likelihood=log_hyperparameter_likelihood)
        
        if need_to_close:
            h5f.close()
            
        return instance

