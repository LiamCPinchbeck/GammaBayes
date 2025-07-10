from .parameter_class import Parameter
import warnings, logging
import numpy as np
import h5py, pickle
import time


from icecream import ic


class ParameterSet(object):
    """
    A collection of Parameter objects, organized for efficient manipulation, retrieval,
    and storage. Supports various input formats for initializing the set, including
    dictionaries, lists, and other ParameterSet instances.
    """

    def __init__(self, parameter_specifications: dict | list | tuple = None, 
                 set_name: str = "UnnamedSet", prior_name: str = None):
        """
        Initializes a ParameterSet with the given parameter specifications.

        Args:
            parameter_specifications (dict | list | ParameterSet | tuple, optional): A dictionary, list, tuple, or another
                ParameterSet instance containing the specifications for initializing the parameter set.
                This can include parameter names, types, prior names, and their specifications or instances.
            set_name (str, optional): Name of the parameter set. Defaults to "UnnamedSet".
            prior_name (str, optional): Name of the prior associated with the parameter set. Defaults to None.
        """

        if parameter_specifications is None:
            parameter_specifications = {}

        self.axes_by_type = {'spectral_parameters':{}, 'spatial_parameters':{}}
        self._dict_of_parameters_by_type = {'spectral_parameters':{}, 'spatial_parameters':{}}
        self._dict_of_parameters_by_name = {}

        try:
            if 'set_name' in parameter_specifications:
                self.set_name = parameter_specifications['set_name']
            else:
                self.set_name = set_name
        except TypeError:
            self.set_name = set_name

        try:
            if 'prior_name' in parameter_specifications:
                self.prior_name = parameter_specifications['prior_name']
            else:
                self.prior_name = prior_name
        except TypeError:
            self.prior_name = prior_name



        if (type(parameter_specifications) == dict):
            

            # Check if parameter_specifications is an empty dict (evaluates to False in python)
            if parameter_specifications:
                try:
                    initial_type = type(list(parameter_specifications.values())[0])
                except IndexError as indexerr:
                    logging.warning(f"An error occurred: {indexerr}")
                    initial_type = None


                # Actually meant to be dict of parameter specifications by type->name->specifications?
                if (initial_type is dict):

                    try:
                        secondary_objects = list(list(parameter_specifications.values())[0].values())[0]
                        secondary_type_is_dict = type(secondary_objects) == dict
                    except IndexError as indexerr:
                        logging.warning(f"An error occurred: {indexerr}")
                        secondary_type_is_dict = False 


                    if secondary_type_is_dict:
                        self.handle_plain_dict_input(parameter_specifications)

                    else:
                        self.handle_name_dict_input(parameter_specifications)
                # Actually meant to be dict of parameter specifications by name->specifications?
                elif (initial_type is Parameter):
                    self.handle_dict_of_param_input(parameter_specifications)


                elif isinstance(parameter_specifications, ParameterSet):
                    self.append(parameter_specifications) 


                else:
                    warnings.warn("Something went wrong when trying to access dictionary values.")

        elif (type(parameter_specifications) == tuple):
            unformatted_parameter_specifications = list(parameter_specifications)
            parameter_specifications = {unformatted_parameter_specifications[0]: unformatted_parameter_specifications[1]}
            
            if type(unformatted_parameter_specifications[1]) == dict:
                self.handle_plain_dict_input(parameter_specifications)

            elif type(unformatted_parameter_specifications[1]) == Parameter:
                self.handle_dict_of_param_input(parameter_specifications)

        elif (type(parameter_specifications)==list):

            # Actually meant to be list of parameter objects?
            if (type(parameter_specifications[0]) is Parameter):
                self.handle_list_param_input(parameter_specifications)

            elif (type(parameter_specifications[0]) is dict):
                self.handle_list_specification_input(parameter_specifications)

        elif isinstance(parameter_specifications, ParameterSet):
            self.append(parameter_specifications) 





    # This method presumes that you have given the specifications in the form of 
            # nested dictionaries of the form
            # prior_name > parameter type > parameter name > parameter specifications
    def handle_plain_dict_input(self, dict_input: dict[dict[dict[dict]]]):
        """
        Handles input in the form of a nested dictionary specifying parameter types, names,
        and specifications, structured by prior name.

        Args:
            dict_input (dict): The input dictionary containing parameter specifications.
        """

        
        for type_key, type_value in dict_input.items():
            for param_name, param_specification in type_value.items():
                parameter = self.create_parameter(param_specification=param_specification, 
                                                    param_name=param_name, 
                                                    type_key=type_key, 
                                                    prior_name=self.prior_name)
                self.append(parameter)

    # This method presumes that you have given the specifications in the form of 
            # nested dictionaries of the form
            # parameter name > parameter specifications
    def handle_name_dict_input(self, dict_input: dict[dict]):
        """
        Handles input where specifications are given in the form of a dictionary keyed
        by parameter name.

        Args:
            dict_input (dict): The input dictionary containing parameter specifications.
        """
        for param_name, single_parameter_specifications in dict_input.items():
            parameter = self.create_parameter(param_specification=single_parameter_specifications, 
                                              param_name=param_name,
                                              prior_name=self.prior_name)
            self.append(parameter)


    # This method presumes that you have given the specifications in the form of 
            # nested dictionaries of the form
            # parameter name > parameter class instances
    def handle_dict_of_param_input(self, dict_input: dict[Parameter]):
        """
        Processes input given as a dictionary of Parameter instances keyed by parameter name.

        Args:
            dict_input (dict): The input dictionary of Parameter instances.
        """
        for param_name, parameter_object in dict_input.items():
            if not('name' in parameter_object):
                parameter_object['name'] = param_name

            if parameter_object['prior_name'] is None:
                parameter['prior_name'] = self.prior_name

            self.append(parameter_object)

    # This method presumes that you have given the specifications in the form of 
            # a list of parameter objects
    def handle_list_param_input(self, list_input : list[Parameter]):
        """
        Accepts a list of Parameter objects and appends them to the set.

        Args:
            list_input (list): The list of Parameter objects.
        """
        for parameter_object in list_input:
            
            if parameter_object['prior_name'] is None:
                parameter['prior_name'] = self.prior_name

            self.append(parameter_object)


    # This method presumes that you have given the specifications in the form of 
            # a list of dictionaries containing the specifications for the parameters
    def handle_list_specification_input(self, list_input: list[Parameter]):
        """
        Handles input given as a list of dictionaries, each containing specifications
        for a parameter.

        Args:
            list_input (list): The list of dictionaries with parameter specifications.
        """
        for parameter_specification in list_input:
            parameter_object = Parameter(parameter_specification)

            if parameter_object['prior_name'] is None:
                parameter['prior_name'] = self.prior_name


            self.append(parameter_object)


    # A kind of filter class for creating parameters for use within the Set
    def create_parameter(self, 
                         param_specification: dict, 
                         param_name: str = None, 
                         type_key: str = None, 
                         prior_name: str = None):
        """
        Creates a Parameter object from given specifications, optionally setting its
        type, name, and associated prior name.

        Args:
            param_specification (dict): Specifications for the parameter.
            param_name (str, optional): Name of the parameter.
            type_key (str, optional): Type of the parameter (e.g., 'spectral_parameters').
            prior_name (str, optional): Name of the prior associated with the parameter.

        Returns:
            Parameter: The created Parameter object.
        """

        parameter = Parameter(param_specification)

        if type_key is not None:
            parameter['parameter_type'] = type_key

        if param_name is not None:
            parameter['name'] = param_name

        if prior_name is not None:
            parameter['prior_name'] = prior_name

        if parameter.get('prior_name') is None:
            parameter['prior_name'] = self.prior_name


        if '_internal_name' not in param_specification:
            _internal_name_temp = str(parameter['name']) +"_"+ time.strftime("%M%S")

            _internal_name_temp += "_"+str(parameter['prior_name'])

            parameter['_internal_name'] =  _internal_name_temp + "_" + str(self._param_counter)

        return parameter
    

    @property
    def _param_counter(self):
        if not hasattr(self, '_parameter_count_idx'):
            self._parameter_count_idx = 0

            return self._parameter_count_idx
        
        self._parameter_count_idx+=1

        return self._parameter_count_idx



    def append(self, parameter: Parameter):
        """
        Appends a Parameter object or another ParameterSet to this set.

        Args:
            parameter (Parameter or ParameterSet): The parameter or parameter set to append.
        """

        if not hasattr(self, "_name_to_internal_name_dict"):
            self._name_to_internal_name_dict = {}
        

        if isinstance(parameter, Parameter):

            param_name = parameter['name']

            parameter['prior_name'] = self.prior_name

            if 'parameter_type' in parameter:
                type_key = parameter['parameter_type']

                if type_key in self.axes_by_type:
                    if 'axis' in parameter:
                        self.axes_by_type[type_key][param_name] = parameter['axis']
                    else:
                        self.axes_by_type[type_key][param_name] = np.nan

            if type_key in self._dict_of_parameters_by_type.keys():
                self._dict_of_parameters_by_type[type_key][parameter._internal_name] = parameter


            self._dict_of_parameters_by_name[parameter._internal_name] = parameter
            self._name_to_internal_name_dict[parameter.name] = parameter._internal_name
            self._name_to_internal_name_dict[parameter._internal_name] = parameter._internal_name


        elif isinstance(parameter, ParameterSet):
            parameters = parameter
            for _internal_param_name, parameter in parameters._dict_of_parameters_by_name.items():

                parameter = self.create_parameter(parameter, 
                                                  param_name=parameter.get("name", None),
                                                  type_key=parameter.get("parameter_type", None),
                                                  prior_name=self.prior_name,
                                                  )
                self.append(parameter)

    def _reset_unique_input_methods(self):
        """
        Resets the unique input methods for the parameters in the set.
        """

        self.parameter_logpdfs = {}

        already_included_parameters = []
        list_internal_param_names = list(self._dict_of_parameters_by_name.keys())
        
        for param_idx, (internal_parameter_name, parameter) in enumerate(self._dict_of_parameters_by_name.items()):


            if not(internal_parameter_name in already_included_parameters):
                hyper_indices_for_param = [param_idx]

                if 'dependent' in parameter:
                    for dependent_parameter_name in parameter['dependent']:
                        hyper_indices_for_param.append(list_internal_param_names.index(dependent_parameter_name))
                        already_included_parameters.append(dependent_parameter_name)


                self.parameter_logpdfs[internal_parameter_name] = [parameter.logpdf, hyper_indices_for_param]



    def __repr__(self):
        """
        Returns a string representation of the ParameterSet.

        Returns:
            str: String representation of the parameter set.
        """
        return str(self._dict_of_parameters_by_name)


    def __len__(self):
        """
        Returns the number of parameters in the set.

        Returns:
            int: The number of parameters.
        """
        return len(self._dict_of_parameters_by_name)


    # Allows the behaviour of set1 + set2 = new combined set
    def __add__(self, other):
        """
        Allows the addition of two ParameterSets, combining their parameters into a new set.

        Args:
            other (ParameterSet): The other ParameterSet to add.

        Returns:
            ParameterSet: A new ParameterSet instance containing parameters from both sets.

        Raises:
            TypeError: If 'other' is not an instance of ParameterSet.
        """
        if not isinstance(other, ParameterSet):
            raise TypeError("Can only add another ParameterSet")

        combined = ParameterSet()
        for param_name, parameter in self._dict_of_parameters_by_name.items():
            combined.append(parameter)

        for param_name, parameter in other._dict_of_parameters_by_name.items():
            combined.append(parameter)  # This will overwrite if param_name already exists

        return combined
    

    # Below three methods are toreplicate straight dictionary behaviour
    def items(self):
        """
        Returns a view of the parameter set's items (parameter name, Parameter object pairs).

        Returns:
            ItemsView: A view of the parameter set's items.
        """
        return self._dict_of_parameters_by_name.items()
    
    def values(self):
        """
        Returns a view of the parameters in the set.

        Returns:
            ValuesView: A view of the Parameter objects in the set.
        """
        return self._dict_of_parameters_by_name.values()
    
    def keys(self):
        """
        Returns a view of the parameter names in the set.

        Returns:
            KeysView: A view of the parameter names.
        """
        return self._dict_of_parameters_by_name.keys()
        

    # Outputs the dictionary of the form used for analysis classes that explore 
        # parameter spaces via scan
    @property
    def scan_format(self):
        """
        Returns a representation of the parameter set formatted for scanning analysis classes.

        Returns:
            dict: A dictionary representation of the parameter set suitable for scanning.
        """
        return self.axes_by_type
    

    @property
    def bounds(self):
        """
        Retrieves the bounds for all parameters in the set.

        Returns:
            list: A list of bounds for the parameters.
        """
        bounds = []
        for param in self._dict_of_parameters_by_name.values():
            bounds.append(param['bounds'])

        return bounds
    
    # Outputs the dictionary of the form used for analysis classes that explore 
        # parameter spaces via stochastic samplers
    @property
    def stochastic_format(self):
        """
        Returns a representation of the parameter set formatted for stochastic sampling analysis classes.

        Returns:
            dict: A dictionary representation of the parameter set suitable for stochastic sampling.
        """
        return self._dict_of_parameters_by_name
    
    @property
    def sampling_format(self):
        """
        Alias for the stochastic_format property.

        Returns:
            dict: A dictionary representation of the parameter set suitable for stochastic sampling.
        """
        return self.stochastic_format
    
    # Method specifically when all parameters are discrete and have a defined 'axis' method
    @property
    def axes(self):
        """
        Returns the axes for all parameters in the set, applicable for discrete parameters.

        Returns:
            list: A list of axes for the parameters.
        """
        return [param['axis'] for param in self._dict_of_parameters_by_name.values()]
    
    # Ex
    @property
    def defaults(self):
        """
        Retrieves the default values for all parameters in the set.

        Returns:
            list: A list of default values for the parameters.
        """
        default_values = []
        for parameter in self._dict_of_parameters_by_name.values():

            try:
                default_values.append(parameter['default_value'])
            except KeyError:
                logging.warn("""One or more parameters have not been given a 
default value. Place nan in position of default""")
                default_values.append(np.nan)
        return default_values
    

    def _fetch_defaults(self, param_type):
        """
        Retrieves the default values for parameters of a specific type.

        Args:
            param_type (str): The type of parameters to retrieve defaults for.

        Returns:
            dict: A dictionary of parameter names and their default values.
        """
        default_values = {}
        for parameter in self._dict_of_parameters_by_type[param_type].values():
            default_values[parameter['name']] = parameter['default_value']
        return default_values   
     
    @property
    def spectral_defaults(self):
        """
        Retrieves the default values for spectral parameters.

        Returns:
            dict: A dictionary of spectral parameter names and their default values.
        """
        return self._fetch_defaults('spectral_parameters')
    

        
    @property
    def spatial_defaults(self):
        """
        Retrieves the default values for spatial parameters.

        Returns:
            dict: A dictionary of spatial parameter names and their default values.
        """
        return self._fetch_defaults('spatial_parameters')
    


    
    # When some sort of iterable behaviour is expected of the class it
        # defaults to using the behaviour of the dict_of_parameters_by_name dict
    def __iter__(self):
        """
        Allows iteration over the ParameterSet, yielding parameter names.

        This method facilitates direct iteration over the parameter set, providing an easy way
        to loop through all parameter names contained within it.

        Returns:
            iterator: An iterator over the names of parameters in the set.
        """
        return iter(self._dict_of_parameters_by_name)



    def pack(self, h5f=None, file_name=None):
        """
        Packs the ParameterSet data into an HDF5 format.

        Returns:
        h5py.File: An HDF5 file object containing the packed data.
        """
        if h5f is None:
            h5f = h5py.File(file_name, 'w-')  # 'None' creates an in-memory HDF5 file object
            
        # Save the order of parameters as an attribute. h5py does not natively support
            # saving groups in a particular order. They're closer to sets than anything.
        param_order = list(self._dict_of_parameters_by_name.keys())
        h5f.attrs['parameter_order'] = param_order
        
        for param_name, parameter_specs in self._dict_of_parameters_by_name.items():
            param_group = h5f.create_group(param_name)
            for spec_attr, spec_attr_val in parameter_specs.items():
                if (spec_attr != 'transform'):
                    if isinstance(spec_attr_val, (np.ndarray, list)):
                        param_group.create_dataset(spec_attr, data=spec_attr_val)
                    elif isinstance(spec_attr_val, (str, tuple, set, int, float)):
                        param_group.attrs[spec_attr] = spec_attr_val
                    elif spec_attr=='prob_model':
                        try:
                            param_group.attrs['prob_model_name'] = spec_attr_val.__name__
                        except:
                            param_group.attrs['prob_model_type'] = str(type(spec_attr_val))
                        
                    else:
                        warnings.warn(f"Could not save {spec_attr} for the parameter {param_name}")

        return h5f
    
    def save(self, file_name):
        """
        Saves the ParameterSet data to an HDF5 file.

        Args:
        file_name (str): The name of the file to save the data to.
        """
        h5f = self.pack(file_name=file_name)
        h5f.close()


    @classmethod
    def load(cls, h5f=None, file_name=None,):
        """
        Loads a ParameterSet from an HDF5 file.

        Args:
            h5f (h5py.File, optional): An HDF5 file object. If None, the file specified by file_name will be opened.
            file_name (str, optional): The name of the file to load the data from.

        Returns:
            ParameterSet: A new ParameterSet instance containing the loaded data.
        """
        new_parameter_set = cls()  # Instantiate a new ParameterSet

        if h5f is None:
            h5f = h5py.File(file_name, 'r')
        
        param_order = list(h5f.attrs['parameter_order'])
        
        for param_name in param_order:
            if param_name in h5f:  # Check if the parameter name exists in the file
                param_group = h5f[param_name]
                parameter_specs = {}
                
                for attr_name in param_group.attrs.keys():
                    if attr_name != 'transform':
                        parameter_specs[attr_name] = param_group.attrs[attr_name]
                
                for dataset_name in param_group.keys():
                    if dataset_name != 'transform':
                        parameter_specs[dataset_name] = param_group[dataset_name][()]
                
                # Create and append the parameter object accordingly
                # This assumes a constructor or method to recreate a parameter from specs
                parameter = Parameter(parameter_specs)
                new_parameter_set._dict_of_parameters_by_name[parameter._internal_name] = parameter

        return new_parameter_set
    
    def save_to_pickle(self, file_name: str):
        """
        Saves the parameter object to an pickle file.

        Args:
            file_name (str): The name of the file to save the parameter data.
        """

        if not(file_name.endswith('.pkl')):
                    file_name = file_name+'.pkl'

        pickle.dump(self, open(file_name,'wb'))


    @classmethod
    def load_from_pickle(cls, file_name: str):
        """
        Loads and initializes a Parameter object from an pickle file.

        Args:
            file_name (str): The name of the HDF5 file from which to load parameter data.

        Returns:
            Parameter: An initialized Parameter object with data loaded from the file.
        """
        if not(file_name.endswith(".pkl")):
            file_name = file_name + ".pkl"

        
        return  pickle.load(open(file_name,'rb'))
    
    def reorder(self, order_list):
        """
        Reorders the parameters in the set based on the provided order list. Can only be used if parameter names within set are unique.

        Args:
            order_list (list): A list of parameter names in the desired order.
        """
        reordered_parameters = {self._name_to_internal_name_dict[name]: self._dict_of_parameters_by_name[self._name_to_internal_name_dict[name]] for name in order_list}
        self._dict_of_parameters_by_name = reordered_parameters

        # Reorder dict_of_parameters_by_type
        for type_key in self._dict_of_parameters_by_type:
            type_parameters = self._dict_of_parameters_by_type[type_key]
            reordered_type_parameters = {name: type_parameters[name] for name in order_list if name in type_parameters}
            self._dict_of_parameters_by_type[type_key] = reordered_type_parameters

        # Reorder axes_by_type
        for type_key in self.axes_by_type:
            type_axes = self.axes_by_type[type_key]
            reordered_axes = {name: type_axes[name] for name in order_list if name in type_axes}
            self.axes_by_type[type_key] = reordered_axes


    def logpdf(self, input):
        """
        Computes the log of the overall sets' probability density function (logPDF) for the given input.

        Args:
            input (array-like): Input values for which to compute the logPDF.

        Returns:
            float: The computed logPDF value.
        """
        return np.sum([logpdf(input[input_indices]) for logpdf, input_indices in self.parameter_logpdfs])
    

    def pdf(self, input):
        """
        Computes the probability density function (PDF) for the given input.

        Args:
            input (array-like): Input values for which to compute the PDF.

        Returns:
            float: The computed PDF value.
        """
        return np.exp(self.logpdf(input=input))
    


    def get(self, keyname, value=None):
        return self._dict_of_parameters_by_name.get(keyname, value=value)




