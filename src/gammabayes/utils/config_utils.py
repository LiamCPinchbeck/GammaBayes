import warnings, yaml, sys, time, pickle, copy
from gammabayes.utils.event_axes import create_axes
import numpy as np


def read_config_file(file_path):
    """
    Reads a configuration file in YAML format and returns its contents.

    Args:
        file_path (str): Path to the configuration file.

    Returns:
        dict: Parsed contents of the configuration file.

    Raises:
        SystemExit: If the file is not found or YAML parsing fails.
    """
    print(f"Configuration File Path: {file_path}\n")
    try:
        with open(file_path, 'r') as file:
            inputs = yaml.safe_load(file)
        return inputs
    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError:
        print(f"Error: Unable to parse YAML in '{file_path}'. Please ensure it is valid.")
        sys.exit(1)
        
        
def check_necessary_config_inputs(input_dict):
    """
    Checks for necessary inputs in the configuration dictionary and prints their values.
    If any necessary input is missing, it assigns default values or raises an exception.

    Args:
        input_dict (dict): Configuration dictionary.

    Raises:
        Exception: If the number of events is not provided.
    """
    

    try:
        print(f"\n\nNumber of events for this script is {input_dict['Nevents']}")
    except:
        raise Exception("Number of events to be simulated/analysed not provided. Add line 'Nevents:  [number value]' to yaml file")
            
    try:
        print(f"Provided identifier is {input_dict['identifier']}")
    except:
        warnings.warn("Identifier not provided. Default value of date in format [yr]_[month]_[day]_[hour]_[minute] will be used.", UserWarning)
        input_dict['identifier'] = time.strftime("%y_%m_%d_%H_%M")
        
    try:
        print(f"Run number is {input_dict['runnumber']}")
    except:
        input_dict['runnumber'] = 1
        
    try:
        print(f"Total number of events is {input_dict['totalevents']}")
    except:
        input_dict['totalevents'] = input_dict['Nevents']
        
    print('\n\n')


def load_hyperparameter_pickle(file_path):
    """
    Loads hyperparameters from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        object: Loaded data from the pickle file.
    """
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def add_event_axes_config(config_dict, energy_axis_true, longitudeaxistrue, latitudeaxistrue, 
    energy_axis, longitudeaxis, latitudeaxis):
    """Takes in config dict and relevant event axes and saves min, max, spacing/resolution
        for each.

    Args:
        config_dict (dict): dict containing run information
        energy_axis_true (array_like):    energy_axis_true for analysis
        longitudeaxistrue (array_like): longitudeaxistrue for analysis
        latitudeaxistrue (array_like):  latitudeaxistrue for analysis
        energy_axis (array_like):        energy_axis for analysis
        longitudeaxis (array_like):     longitudeaxis for analysis
        latitudeaxis (array_like):      latitudeaxis for analysis

    Returns:
        dict: config file with added information
    """
    config_dict['true_energy_min']                = energy_axis_true.min()
    config_dict['true_energy_max']                = energy_axis_true.max()
    config_dict['true_energy_bins_per_decade']    = (len(energy_axis_true)-1)/np.log10(energy_axis_true).ptp()

    config_dict['true_spatial_res']               = np.diff(longitudeaxistrue)[0]
    config_dict['true_longitude_min']             = longitudeaxistrue.min()
    config_dict['true_longitude_max']             = longitudeaxistrue.max()
    config_dict['true_latitude_min']              = latitudeaxistrue.min()
    config_dict['true_latitude_max']              = latitudeaxistrue.max()


    config_dict['recon_energy_min']               = energy_axis.min()
    config_dict['recon_energy_max']               = energy_axis.max()
    config_dict['recon_energy_bins_per_decade']   = (len(energy_axis)-1)/np.log10(energy_axis).ptp()

    config_dict['recon_spatial_res']              = np.diff(longitudeaxistrue)[0]
    config_dict['recon_longitude_min']            = longitudeaxis.min()
    config_dict['recon_longitude_max']            = longitudeaxis.max()
    config_dict['recon_latitude_min']             = latitudeaxis.min()
    config_dict['recon_latitude_max']             = latitudeaxis.max()

    return config_dict




def save_config_file(config_dict, file_path):
    """
    Saves the configuration dictionary to a YAML file.

    Args:
        config_dict (dict): Configuration dictionary to be saved.
        file_path (str): Path to the YAML file.
    """
    with open(file_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)


def create_true_axes_from_config(config_dict):
    """
    Creates true event axes based on the configuration dictionary.

    Args:
        config_dict (dict): Configuration dictionary containing axis parameters.

    Returns:
        tuple: Created true event axes.
    """

    return create_axes(config_dict['true_energy_min'], config_dict['true_energy_max'], 
                     config_dict['true_energy_bins_per_decade'], config_dict['true_spatial_res'], 
                     config_dict['true_longitude_min'], config_dict['true_longitude_max'],
                     config_dict['true_latitude_min'], config_dict['true_latitude_max'],
                     custom_print_str="__________ TRUE EVENT AXIS PARAMETERS __________\n")

def create_recon_axes_from_config(config_dict):
    """
    Creates reconstructed event axes based on the configuration dictionary.

    Args:
        config_dict (dict): Configuration dictionary containing axis parameters.

    Returns:
        tuple: Created reconstructed event axes.
    """
    return create_axes(config_dict['recon_energy_min'], config_dict['recon_energy_max'], 
                     config_dict['recon_energy_bins_per_decade'], config_dict['recon_spatial_res'], 
                     config_dict['recon_longitude_min'], config_dict['recon_longitude_max'],
                     config_dict['recon_latitude_min'], config_dict['recon_latitude_max'],
                     custom_print_str="__________ MEASURED/RECONSTRUCTED EVENT AXIS PARAMETERS __________\n")


def construct_parameter_axes(construction_config):
    """
    Constructs parameter axes based on the construction configuration.

    Args:
        construction_config (dict or list): Configuration dictionary or list for constructing parameter axes.

    Raises:
        Exception: If invalid configuration or bounds are provided.

    Returns:
        np.ndarray: Constructed parameter axes.
    """

    # Copying the dict so I don't accidentally overwrite the original
    construction_config_copy = copy.deepcopy(construction_config)
    if type(construction_config_copy)==list:
        return np.asarray(construction_config_copy)
    

    elif construction_config_copy['spacing']=='custom':
        return np.load(construction_config_copy['custom_bins_filepath'])


    elif construction_config_copy['spacing']=='linear':

        if type(construction_config_copy['bounds'])==list:
            return np.linspace(float(construction_config_copy['bounds'][0]), 
                               float(construction_config_copy['bounds'][1]), 
                               int(construction_config_copy['nbins']))

        
        elif construction_config_copy['bounds']=='event_dynamic':
            if ('num_events' in construction_config_copy) and ('true_val' in construction_config_copy):
                num_events  = float(construction_config_copy['num_events'])
                true_val    = construction_config_copy['true_val']

                if 'dynamic_multiplier' in construction_config_copy:
                    dynamic_multiplier = float(construction_config_copy['dynamic_multiplier'])
                else:
                    dynamic_multiplier = 10
                

                lower_bound     = true_val - dynamic_multiplier/np.sqrt(num_events)
                upper_bound     = true_val + dynamic_multiplier/np.sqrt(num_events)

                if 'absolute_bounds' in construction_config_copy:
                    absolutes = [float(construction_config_copy['absolute_bounds'][0]), float(construction_config_copy['absolute_bounds'][1])]
                else:
                    absolutes = [-np.inf, np.inf]

                if lower_bound < absolutes[0]:
                    lower_bound = absolutes[0]

                if upper_bound > absolutes[1]:
                    upper_bound = absolutes[1]
                
                return np.linspace(lower_bound, upper_bound, int(construction_config_copy['nbins']))

            else:
                if 'absolute_bounds' in construction_config_copy:
                    warnings.warn("Number of events or true value not given for event dynamic bound setting. Defaulting to absolute bounds")
                    construction_config_copy['bounds'] = construction_config_copy['absolute_bounds']

                    return construct_parameter_axes(construction_config_copy)
                else:
                    raise Exception("Number of events or true value, and absolute bounds not given for event dynamic bounds. Please check inputs.")
        else:
            raise Exception("Bounds must either be a list, [lower bound, upper bound] or the string 'event_dynamic'. ")
                

    
    elif construction_config_copy['spacing']=='logspace':

        if type(construction_config_copy['bounds'])==list:
            return np.logspace(np.log10(float(construction_config_copy['bounds'][0])), 
                               np.log10(float(construction_config_copy['bounds'][1])), 
                               int(construction_config_copy['nbins']))
        

        elif construction_config_copy['bounds']=='event_dynamic':
            if 'num_events' in construction_config_copy:
                num_events = float(construction_config_copy['num_events'])

                if 'dynamic_multiplier' in construction_config_copy:
                    dynamic_multiplier = float(construction_config_copy['dynamic_multiplier'])
                else:
                    dynamic_multiplier = 10
                

                lower_bound = np.log10(construction_config_copy['true_val']) - dynamic_multiplier/np.sqrt(num_events)
                upper_bound = np.log10(construction_config_copy['true_val']) + dynamic_multiplier/np.sqrt(num_events)

                if 'absolute_bounds' in construction_config_copy:
                    absolutes = [float(construction_config_copy['absolute_bounds'][0]), float(construction_config_copy['absolute_bounds'][1])]
                else:
                    absolutes = [-np.inf, np.inf]

                if lower_bound < np.log10(absolutes[0]):
                    lower_bound = np.log10(absolutes[0])

                if upper_bound > np.log10(absolutes[1]):
                    upper_bound = np.log10(absolutes[1])
                
                return np.logspace(lower_bound, upper_bound, int(construction_config_copy['nbins']))

            else:
                if 'absolute_bounds' in construction_config_copy:
                    warnings.warn("Number of events not given for event dynamic bound setting. Defaulting to absolute bounds")
                    construction_config_copy['bounds'] = construction_config_copy['absolute_bounds']

                    return construct_parameter_axes(construction_config_copy)
                else:
                    raise Exception("Neither number of events or absolute bounds given for event dynamic bounds.")
    
        else:
            raise Exception("Bounds must either be a list, [lower bound, upper bound] or the string 'event_dynamic'. ")

    else:
        raise Exception("Unknown spacing condition given. Please check 'spacing' keyword in dictionary given. Must be 'logspace', 'linear' or 'custom'.")



def iterative_parameter_axis_construction(scan_specification_dict):
    """
    Constructs parameter axes iteratively based on the scan specification dictionary.

    Args:
        scan_specification_dict (dict): Dictionary specifying the scan configuration.

    Returns:
        dict: Dictionary with constructed parameter axes.
    """



    scan_dict = copy.deepcopy(scan_specification_dict)
    for prior_scan_specifications in scan_specification_dict:
        for parameter_type_scan_dict in scan_specification_dict[prior_scan_specifications]:
            for parameter in scan_specification_dict[prior_scan_specifications][parameter_type_scan_dict]:
                temp_single_param_scan_dict = scan_specification_dict[prior_scan_specifications][parameter_type_scan_dict][parameter]
                scan_dict[prior_scan_specifications][parameter_type_scan_dict][parameter] = construct_parameter_axes(temp_single_param_scan_dict)


    return scan_dict