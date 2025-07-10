import numpy as np, warnings, dynesty, logging, time

from gammabayes.hyper_inference.core.utils import _handle_parameter_specification
from gammabayes.hyper_inference.core.mixture_tree import MTree, MTreeNode
from gammabayes.samplers.sampler_utils import ResultsWrapper
from gammabayes import ParameterSet, ParameterSetCollection
from gammabayes.core.core_utils import update_with_defaults
from itertools import islice

import h5py

class ScanOutput_StochasticTreeMixturePosterior(object):
    """
    A class designed to handle stochastic exploration of posterior distributions for general 
    mixture fractions using the Dynesty nested sampling package and the "Tree" GammaBayes class.

    Attributes:
        prior_parameter_specifications (list | dict | ParameterSet): Specifications for prior parameters.
        mixture_tree (MTree): Tree handling the layout/structure of the mixture.
        mixture_parameter_specifications (ParameterSet): Specifications for mixture parameters.
        log_nuisance_marg_results (np.array): Logarithm of marginal results for the parameters.
        log_nuisance_marg_regularisation (float): Regularisation parameter for nuisance marginalisation.
        num_priors (int): The number of prior distributions provided.
        num_events (int): The number of events in the log marginal results.
        mixture_bounds (list): Bounds for the mixture parameters.
        num_mixes (int): The number of mixture components.
        hyperparameter_axes (list): Axes for hyperparameters, derived from prior parameter specifications.
        index_to_hyper_parameter_info (dict): Mapping of hyperparameter indices to their specifications.
        ndim (int): The number of dimensions for the sampler, including mixture components and hyperparameters.
        sampler (dynesty.NestedSampler): The Dynesty sampler object used for exploration.
    """
    def __init__(self, 
                log_nuisance_marg_results: list[np.ndarray]|np.ndarray,
                mixture_tree: MTree,
                mixture_parameter_specifications:list | dict | ParameterSet,
                prior_parameter_specifications: list | dict | ParameterSet = None,
                observational_prior_names: list=None,
                shared_parameters: list | dict | ParameterSet = None,
                parameter_meta_data: dict = None,
                event_weights:np.ndarray=None,
                log_nuisance_marg_regularisation: float = 0., # Argument for consistency between classes
                _sampler_results={}):
        """
        Initializes the ScanOutput_StochasticTreeMixturePosterior class with necessary parameters and configurations.

        Args:
            log_nuisance_marg_results (list[np.ndarray] | np.ndarray): Logarithm of marginal results for the parameters.
            mixture_tree (MTree): Tree handling the layout/structure of the mixture.
            mixture_parameter_specifications (list | dict | ParameterSet): Specifications for mixture parameters.
            log_nuisance_marg_regularisation (float, optional): Regularisation parameter for nuisance marginalisation.
            prior_parameter_specifications (list | dict | ParameterSet, optional): Specifications for prior parameters.
            shared_parameters (list | dict | ParameterSet, optional): Shared parameters across the model. Defaults to {}.
            parameter_meta_data (dict, optional): Metadata for the parameters. Defaults to {}.
            _sampler_results (dict, optional): Initial results for the sampler. Defaults to {}.
        """
        if shared_parameters is None:
            shared_parameters = {}
        if parameter_meta_data is None:
            parameter_meta_data = {}

        self.prior_parameter_specifications = _handle_parameter_specification(
            prior_parameter_specifications,
            _no_required_num=True
        )
        self.mixture_tree = mixture_tree
        self.shared_parameters = shared_parameters



        self.mixture_parameter_specifications = ParameterSet(mixture_parameter_specifications)

        # Making sure that the order of the mixture parameters is the same as the tree layout
        mixture_param_order = mixture_tree.nodes.copy()
        del mixture_param_order['root']
        self.mixture_parameter_specifications.reorder(list(mixture_param_order))


        self.mixture_bounds         = self.mixture_parameter_specifications.bounds


        self.parameter_set_collection = ParameterSetCollection(
            parameter_sets            = self.prior_parameter_specifications,
            mixture_parameter_set     = self.mixture_parameter_specifications,
            shared_parameters         = self.shared_parameters, 
            parameter_meta_data       = {},
            observational_prior_names = observational_prior_names,
            mixture_tree              = self.mixture_tree,
            collection_name = 'ScanOutput_Stochastic_MixtureFracPosterior parameter set collection'
        )

        # self.hyperparameter_axes    = [param_set.axes_by_type for param_set in self.prior_parameter_specifications]

        self.num_shared_params = len(self.shared_parameters)



        self.log_nuisance_marg_results = log_nuisance_marg_results
        # self.num_priors     = len(self.log_nuisance_marg_results)
        # self.num_events     = log_nuisance_marg_results[0].shape[0]


#         if not(len(self.mixture_bounds) +1 != len(log_nuisance_marg_results)):
#             warnings.warn("""Number of mixtures +1 does not match number of 
# priors indicated in log_nuisance_marg_results. Assigning min=0 and max=1 for remaining mixtures.""")
            
#             assert len(log_nuisance_marg_results[0])>len(self.mixture_bounds) +1

#             for missing_mix_idx in range(len(log_nuisance_marg_results)-(len(self.mixture_bounds) +1)):
#                 self.mixture_bounds.append([0., 1.])

        # -1 is to not count the root node, which is always 1.
        self.num_mixes   = len(self.mixture_tree.nodes)-1 -sum([len(equiv_nodes)-1 for equiv_nodes in self.mixture_tree.equivalent_weight_node_mappings])

        if event_weights is None:
            event_weights = np.ones(shape=(len(log_nuisance_marg_results[0]),))
        
        self.event_weights = event_weights

        self.results = _sampler_results

        self.ndim = len(self.parameter_set_collection.hyper_param_index_to_info_dict)


        # self.set_hyper_axis_info()





    # def set_hyper_axis_info(self):
    #     """
    #     Sets the hyperparameter axis information.

    #     This method initializes the number of hyperparameter axes and creates a mapping of hyperparameter indices
    #     to their specifications.
    #     """

    #     # Counter for hyperparameter axes, mostly for 'index_to_hyper_parameter_info'
    #     hyper_idx = 0

        # # Creating components of mixture for each prior
        # self.num_hyper_axes         = 0
        # index_to_hyper_parameter_info = {}

        # for prior_idx, prior_hyper_axes in enumerate(self.hyperparameter_axes):
        #     idx_for_prior = 0

        #     update_with_defaults(prior_hyper_axes, {'spectral_parameters': {}, 'spatial_parameters':{}})

        #     self.num_hyper_axes+=len(prior_hyper_axes['spectral_parameters'])
        #     self.num_hyper_axes+=len(prior_hyper_axes['spatial_parameters'])

        #     for hyp_name, hyp_axis in prior_hyper_axes['spectral_parameters'].items():
        #         # print('hyper_idx: ', prior_idx, 'spectral_parameters', hyp_name, hyper_idx, idx_for_prior)
        #         index_to_hyper_parameter_info[hyper_idx] = [prior_idx, 'spectral_parameters', hyp_name, hyp_axis, idx_for_prior]
        #         idx_for_prior+1
        #         hyper_idx+=1

        #     for hyp_name, hyp_axis in prior_hyper_axes['spatial_parameters'].items():
        #         # print('hyper_idx: ', prior_idx, 'spatial_parameters', hyp_name, hyper_idx, idx_for_prior)
        #         index_to_hyper_parameter_info[hyper_idx] = [prior_idx, 'spatial_parameters', hyp_name, hyp_axis, idx_for_prior]
        #         idx_for_prior+1
        #         hyper_idx+=1

        # self.index_to_hyper_parameter_info = index_to_hyper_parameter_info





        
    def prior_transform(self, u:float|np.ndarray):
        """
        Transforms unit cube values to prior parameter values based on self.parameter_set_collection prior_transform.

        Args:
            u (float | np.ndarray): Unit cube values.

        Returns:
            np.ndarray: Transformed prior parameter values.
        """


        unitcube = np.squeeze(self.parameter_set_collection.prior_transform(u))

        return unitcube
    

    
    def single_event_loglike(self, inputs:list|np.ndarray, log_nuisance_marg_results:list[np.ndarray]=None):
        """
        Calculates the log likelihood of the given inputs.

        Args:
            inputs (list | np.ndarray): Input values for the parameters.
            log_nuisance_marg_results (list[np.ndarray], optional): Logarithm of marginal results for the parameters.

        Returns:
            float: The log likelihood value.
        """

        if log_nuisance_marg_results is None:
            log_nuisance_marg_results = self.log_nuisance_marg_results
        

        # Extract values of parameters
        mixture_fractions = inputs[:self.num_mixes]

        self.mixture_tree.overwrite(mixture_fractions)


        mixture_weights     = list(self.mixture_tree.leaf_values.values())


        # print("mixture_fractions: ", mixture_fractions)
        # print("mixture_weights: ", mixture_weights)
        
        # shared_parameters = inputs[:, self.num_mixes:self.num_mixes+self.num_shared_params]
        # unique_parameters = inputs[:, self.num_mixes+self.num_shared_params:]

        # Generate slices of log nuisance parameter marginalised matrices



        log_nuisance_param_matrix_slices = [[] for _ in range(len(log_nuisance_marg_results))]


        for hyper_param_idx, hyper_param_info in islice(self.parameter_set_collection.hyper_param_index_to_info_dict.items(), self.num_mixes, None):
            # print(hyper_param_idx, log_nuisance_param_matrix_slices, inputs[hyper_param_idx])
            
            prior_axes = hyper_param_info['prior_param_axes']
            prior_param_value = inputs[hyper_param_idx]


            for prior_idx in hyper_param_info['prior_identifiers']:
                axis_idx = hyper_param_info['prior_to_axis_dict'][prior_idx]
                log_nuisance_param_slice =  np.abs(prior_param_value-prior_axes[axis_idx]).argmin()
                # log_nuisance_param_slice = slice_argmin.flatten()[0]


                log_nuisance_param_matrix_slices[prior_idx].append(log_nuisance_param_slice)


        ln_like = -np.inf
        for prior_idx, mixture_weight in enumerate(mixture_weights,):
            ln_component = np.log(mixture_weight)

            ln_marg_results_for_prior = log_nuisance_marg_results[prior_idx]

            ln_results_slice = log_nuisance_param_matrix_slices[prior_idx]

            ln_comp_marg_comp = ln_marg_results_for_prior[:, *ln_results_slice]

            ln_component += ln_comp_marg_comp


            ln_like = np.logaddexp(ln_like, np.squeeze(ln_component))

        return ln_like
    
    def ln_likelihood(self, inputs:list|np.ndarray, log_nuisance_marg_results:list[np.ndarray]=None):

        return np.dot(self.single_event_loglike(inputs=inputs, log_nuisance_marg_results=log_nuisance_marg_results), self.event_weights)




    # Note: When allocating live points it's lower than the norm due to the discrete nature of the prior parameters
    def initiate_exploration(self, **kwargs):
        """
        Initiates the exploration process by setting up the Dynesty sampler with the class's likelihood and prior transform functions.

        Args:
            **kwargs: Keyword arguments to be passed to the Dynesty sampler.

        Returns:
            dynesty.NestedSampler: The configured Dynesty sampler object.
        """

        self.sampler = dynesty.NestedSampler(loglikelihood=self.ln_likelihood, 
                                               prior_transform=self.prior_transform, 
                                               ndim=self.ndim, 
                                               **kwargs)
        
        return self.sampler
    
    def run_exploration(self,*args, **kwargs):
        """
        Runs the nested sampling exploration process using the initialized Dynesty sampler.

        Args:
            *args: Positional arguments to be passed to the `run_nested` method of Dynesty sampler.
            **kwargs: Keyword arguments to be passed to the `run_nested` method of Dynesty sampler.

        Returns:
            dict: A dictionary containing the results of the nested sampling run.
        """
        self.run_nested_output = self.sampler.run_nested(*args, **kwargs)
        return self.run_nested_output
    





    def _pack_data(self, h5f=None, file_name:str=None, save_log_marg_results:bool=True):
        """
        Private method that packs the class data into an HDF5 format.

        Equivalent to the public method for use in sub-classes.

        Args:
            h5f (h5py.File): An open HDF5 file object for writing data.
            file_name (str): The name of the HDF5 file to create or write to if `h5f` is None.
            save_log_marg_results (bool): Whether to save log marginal results.

        Returns:
            h5py.File: The HDF5 file object containing the packed data.
        """

        if h5f is None:
            h5f = h5py.File(file_name, 'w-')

        # Ensure mixture_parameter_specifications is properly handled if it's a ParameterSet
        if isinstance(self.mixture_parameter_specifications, ParameterSet):
            # Assuming ParameterSet has a pack method to handle its serialization
            mixture_param_group = h5f.create_group("mixture_parameter_specifications")
            self.mixture_parameter_specifications.pack(mixture_param_group)
        
        # Save log_nuisance_marg_results
        prior_parameters_group = h5f.create_group("prior_parameter_specifications")
        for prior_idx, prior_parameters in enumerate(self.prior_parameter_specifications):
            single_prior_parameters_group = prior_parameters_group.create_group(str(prior_idx))
            single_prior_parameters_group = prior_parameters.pack(single_prior_parameters_group)


        
        # Save log_nuisance_marg_results
        if save_log_marg_results:
            log_nuisance_marg_results_group = h5f.create_group("log_nuisance_marg_results")
            for result_idx, result in enumerate(self.log_nuisance_marg_results):
                log_nuisance_marg_results_group.create_dataset(str(result_idx), data=result)
        else:
            log_nuisance_marg_results_group = h5f.create_group("log_nuisance_marg_results")
            log_nuisance_marg_results_group.create_dataset(str(0), data=np.array([0.]))

        if hasattr(self.sampler, 'results'):
            sampler_group=  h5f.create_group('sampler_results')
            results = self.sampler.results
            
            # Save samples
            if hasattr(results, 'samples'):
                sampler_group.create_dataset('samples', data=np.array(results.samples))
            
            # Save log weights
            if hasattr(results, 'logwt'):
                sampler_group.create_dataset('logwt', data=np.array(results.logwt))
            
            # Save log likelihoods
            if hasattr(results, 'logl'):
                sampler_group.create_dataset('logl', data=np.array(results.logl))
            
            # Save evidence information, if available
            if hasattr(results, 'logz'):
                sampler_group.create_dataset('logz', data=np.array(results.logz))
            
            if hasattr(results, 'logzerr'):
                sampler_group.create_dataset('logzerr', data=np.array(results.logzerr))

            if hasattr(results, 'information'):
                sampler_group.create_dataset('information', data=np.array(results.information))

            if hasattr(results, 'nlive'):
                sampler_group.attrs['nlive'] = int(results.nlive)

            if hasattr(results, 'niter'):
                sampler_group.attrs['niter'] = int(results.niter)

            if hasattr(results, 'eff'):
                sampler_group.attrs['eff'] = float(results.niter)

        return h5f
    
    def pack_data(self, h5f=None, file_name:str=None, save_log_marg_results:bool=False):
        """
        Packs the class data into an HDF5 format.

        Args:
            h5f (h5py.File, optional): An open HDF5 file object for writing data.
            file_name (str, optional): The name of the HDF5 file to create or write to if `h5f` is None.
            save_log_marg_results (bool, optional): Whether to save log marginalisation results.

        Returns:
            h5py.File: The HDF5 file object containing the packed data.
        """



        return self._pack_data(h5f=h5f, file_name=file_name, save_log_marg_results=save_log_marg_results)
    

    def save(self, file_name:str):
        """
        Saves the class data to an HDF5 file.

        Args:
            file_name (str): The name of the file to save the data to.
        """
        h5f = self.pack_data(file_name=file_name)
        h5f.close()


    @property
    def results(self):
        """
        Retrieves the results from the sampler.

        Returns:
            dict: The results from the sampler.
        """

        if self._results == {}:
            self._results = self.sampler.results
        return self._results
    
    @results.setter
    def results(self, value):
        """
        Sets the results for the sampler.

        Args:
            value (dict): The results to set.
        """
        self._results = value


    @classmethod
    def load(cls, h5f=None, file_name:str=None, log_nuisance_marg_results:list[np.ndarray]=[]):
        """
        Loads the class data from an HDF5 file.

        Args:
            h5f (h5py.File, optional): An open HDF5 file object for reading data.
            file_name (str, optional): The path to the HDF5 file to load.
            log_nuisance_marg_results (list[np.ndarray], optional): Logarithm of marginal results for the parameters.

        Returns:
            ScanOutput_StochasticTreeMixturePosterior: An instance of the class reconstructed from the file.
        """
        if isinstance(h5f, str):
            file_name = h5f
            h5f=None
        
        if (h5f is None):
            if file_name is None:
                raise ValueError("Either an h5py object or a file name must be provided.")
            # Open the file to get the h5py object
            h5f = h5py.File(file_name, 'r')            

        # Load mixture_parameter_specifications
        if "mixture_parameter_specifications" in h5f:
            mixture_param_group = h5f["mixture_parameter_specifications"]
            # Assuming ParameterSet class has a corresponding load or similar method
            mixture_parameter_specifications = ParameterSet.load(mixture_param_group)
        else:
            mixture_parameter_specifications = None
        
        # Load prior_parameter_specifications
        prior_parameter_specifications = []
        if "prior_parameter_specifications" in h5f:
            prior_parameters_group = h5f["prior_parameter_specifications"]
            for prior_idx in sorted(prior_parameters_group, key=int):
                prior_group = prior_parameters_group[str(prior_idx)]
                # Assuming ParameterSet.load is capable of handling the loading process
                prior_parameters = ParameterSet.load(prior_group)
                prior_parameter_specifications.append(prior_parameters)
            

        _sampler_results = {}
        if 'sampler_results' in h5f:
            sampler_group = h5f['sampler_results']  # Access the group where the sampler data is stored
            
            # Load samples
            if 'samples' in sampler_group:
                _sampler_results['samples'] = np.array(sampler_group['samples'])
            
            # Load log weights
            if 'logwt' in sampler_group:
                _sampler_results['logwt'] = np.array(sampler_group['logwt'])
            
            # Load log likelihoods
            if 'logl' in sampler_group:
                _sampler_results['logl'] = np.array(sampler_group['logl'])
            
            # Load evidence information, if available
            if 'logz' in sampler_group:
                _sampler_results['logz'] = np.array(sampler_group['logz'])
            if 'logzerr' in sampler_group:
                _sampler_results['logzerr'] = np.array(sampler_group['logzerr'])

            if 'information' in sampler_group:
                _sampler_results['information'] = np.array(sampler_group['information'])

            if 'nlive' in sampler_group:
                _sampler_results['nlive'] = int(sampler_group['nlive'])

            if 'niter' in sampler_group:
                _sampler_results['niter'] = int(sampler_group['niter'])

            if 'eff' in sampler_group:
                _sampler_results['eff'] = float(sampler_group['eff'])

        _sampler_results = ResultsWrapper(_sampler_results)
        
        
        # Reconstruct the class instance
        instance = cls(
            log_nuisance_marg_results=log_nuisance_marg_results,
            mixture_parameter_specifications=mixture_parameter_specifications,
            prior_parameter_specifications=prior_parameter_specifications,  # Assuming your __init__ can handle this
            _sampler_results=_sampler_results,
        )
            
        return instance

    
        