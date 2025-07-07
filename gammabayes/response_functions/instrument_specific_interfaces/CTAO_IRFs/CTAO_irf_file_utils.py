from gammabayes import haversine, resources_dir
import numpy as np
from astropy import units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
import os
import zipfile
from icecream import ic

def find_file_in_parent(parent_folder:str, end_of_filename:str):
    """
    Searches for a file with a specific ending in a parent folder.

    Args:
        parent_folder (str): The parent directory to search in.
        end_of_filename (str): The file name ending to search for.

    Returns:
        str | None: The full path to the file if found, otherwise None.
    """

    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(end_of_filename):
                return os.path.join(root, file)
    return None



def find_ctao_irf_file_path(zenith_angle:int=20, hemisphere:str='South', prod_vers=5, irf_time_seconds = 18000, subarray:str=None):
    """
    Finds the file path for the CTA IRF file based on given parameters.

    Args:
        zenith_angle (int, optional): Zenith angle. Defaults to 20.
        hemisphere (str, optional): Hemisphere ('North' or 'South'). Defaults to 'South'.
        prod_vers (int, optional): Production version (3 or 5). Defaults to 5.
        subarray (str, optional): Subarray type (e.g., 'SST', 'MST', 'LST'). Defaults to None.

    Raises:
        Exception: If invalid input values are provided.

    Returns:
        tuple[str, str]: The IRF version and the file path.
    """

    prod5_variations = [5,'prod5', '5', 'prod5-v0.1']
    prod3_variations = [3,'prod3', '3', '3b', 'prod3b']


    if prod_vers in prod5_variations:
        parent_irf_folder = resources_dir+"/CTAO_irf_fits_files/prod5/"
        prod_vers = 'prod5-v0.1'
    elif prod_vers in prod3_variations:
        parent_irf_folder = resources_dir+"/CTAO_irf_fits_files/prod3b/"
        prod_vers = 'prod3b'
    else:
        raise Exception(f"Invalid irf version given must, be one of {prod5_variations} or {prod3_variations}.")


    hemisphere_south_variations = ['s','south']
    hemisphere_north_variations = ['n','north']

    if hemisphere.lower() in hemisphere_south_variations:
        hemisphere = 'South'
    elif hemisphere.lower() in hemisphere_north_variations:
        hemisphere = 'North'
    else:
        raise Exception(f"Invalid hemisphere input given, must be one of {hemisphere_south_variations} or {hemisphere_north_variations}.")
    
    possible_zenith_angles = [20, 40, 60]

    if int(zenith_angle) in possible_zenith_angles:
        zenith_angle = str(int(zenith_angle))
    else:
        raise Exception(f"Invalid zenith angle given, must be one of {possible_zenith_angles}.")
    

    possible_subarrays = ['sstsubarray', 'mstsubarray', 'lstsubarray', 'sst', 'mst', 'lst']
    formatted_subarray_types = ['SSTSubArray', 'MSTSubArray', 'LSTSubArray', 'SSTSubArray', 'MSTSubArray', 'LSTSubArray']

    if not(subarray is None):
        if subarray.lower() in possible_subarrays:
            subarray = formatted_subarray_types[possible_subarrays.index(subarray.lower())]
            subarray = subarray+'-'
        else:
            raise Exception(f'Invalid sub array type given. Must be one of {formatted_subarray_types}.')
    else:
        subarray = ''

    irf_times = np.array([1800, 18000, 180000])

    time = irf_times[np.abs(irf_time_seconds-irf_times).argmin()]



    if prod_vers == 'prod5-v0.1':
        untarred_parent_folder= parent_irf_folder+f"CTA-Performance-{prod_vers}-{hemisphere}-{subarray}{zenith_angle}deg.FITS"
        end_of_filename = f'{time}s-v0.1.fits.gz'

        filename = find_file_in_parent(untarred_parent_folder, end_of_filename)

        if filename:
            return 'prod5', filename
        else:
            print("Stem file has not been un-tarred.")
            #TODO: Unzip irf files in some sort of setup function on install
            import tarfile 
            
            # open file 
            try:
                file = tarfile.open(untarred_parent_folder+'.tar.gz') 
                # extracting file 
                file.extractall(untarred_parent_folder)

                file.close() 

                filename = find_file_in_parent(untarred_parent_folder, end_of_filename)

                if filename:
                    return 'prod5', filename
                else:
                    raise Exception('Found tarred file but could not extract sub-file.')
            except:
                raise Exception("Could not find prod5 irf file for the given inputs.")
    elif prod_vers == 'prod3b':

        unzipped_parent_folder = parent_irf_folder+f"{hemisphere}_z{zenith_angle}_50h"

        filename = find_file_in_parent(unzipped_parent_folder, '.fits')

        if filename:
            return 'prod3b', filename
        else:
            zip_filename = unzipped_parent_folder+'.zip'

            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                dirname = os.path.dirname(unzipped_parent_folder)
                zip_ref.extractall(dirname)

            filename = find_file_in_parent(unzipped_parent_folder, '.fits')

            print(filename)

            if filename:
                return 'prod3b', filename
            else:
                raise Exception('Found ziiped file but could not extract sub-file.')
