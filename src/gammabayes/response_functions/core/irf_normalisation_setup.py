# If running from main file, the terminal format should be $ python -m gammabayes.utils.default_file_setup 1 1 1
    # If you're running from a script there shouldn't be any issues as setup is just a func
from gammabayes.utils import iterate_logspace_integration
# from gammabayes.likelihoods.irfs.prod5.gammapy_wrappers import log_edisp, log_psf
from gammabayes import resources_dir

from tqdm import tqdm
import numpy as np
from astropy import units as u
from scipy import special
from scipy.integrate import simpson as simps
import os, sys
from astropy.units import Quantity
from numpy import ndarray


def irf_norm_setup(
        log_psf:callable, log_edisp:callable,
        energy_true_axis:ndarray[Quantity], energy_recon_axis:ndarray[Quantity], 
          longitudeaxistrue:ndarray[Quantity], longitudeaxis:ndarray[Quantity], 
          latitudeaxistrue:ndarray[Quantity], latitudeaxis:ndarray[Quantity],
          save_directory:str = '', 
          save_results:bool=False):
    """
    Produces default IRF normalization matrices.

    Args:
        energy_true_axis (array[Quantity]): Discrete true energy values of CTA event data.
        energy_recon_axis (array[Quantity]): Discrete measured energy values of CTA event data.
        longitudeaxistrue (array[Quantity]): Discrete true FOV longitude values of CTA event data.
        longitudeaxis (array[Quantity]): Discrete measured FOV longitude values of CTA event data.
        latitudeaxistrue (array[Quantity]): Discrete true FOV latitude values of CTA event data.
        latitudeaxis (array[Quantity]): Discrete measured FOV latitude values of CTA event data.
        save_directory (str, optional): Path to save results. Defaults to resources_dir.
        log_psf (callable, optional): Function representing the log point spread function for the CTA. Defaults to log_psf.
        log_edisp (callable, optional): Function representing the log energy dispersion for the CTA. Defaults to log_edisp.
        save_results (bool, optional): Whether to save the results. Defaults to False.

    Returns:
        tuple: psfnorm and edispnorm normalization matrices.
    """

    
    if save_results:
        print(f"Save directory is {save_directory}")


    psfnorm = []
    for energy_val in tqdm(energy_true_axis, desc='Setting up psf normalisation', ncols=80):
        psflogerow = []
        for lonval in longitudeaxistrue:
            energy_true_axis_mesh, longitude_axis_true_mesh, latitude_axis_true_mesh, longitude_axis_mesh, latitude_axis_mesh  = np.meshgrid(energy_val,
                                                                                                                                    lonval, 
                                                                                                                                    latitudeaxistrue, 
                                                                                                                                    longitudeaxis, 
                                                                                                                                    latitudeaxis, indexing='ij')

            psfvals = log_psf(longitude_axis_mesh.flatten(), latitude_axis_mesh.flatten(), 
                                energy_true_axis_mesh.flatten(), 
                                longitude_axis_true_mesh.flatten(), 
                                latitude_axis_true_mesh.flatten(),).reshape(energy_true_axis_mesh.shape)
            psfnormvals = iterate_logspace_integration(np.squeeze(psfvals), axes=[longitudeaxis, latitudeaxis], axisindices=[1,2])
            
            psflogerow.append(psfnormvals)
        psfnorm.append(psflogerow)

    edispnorm = []
    for energy_val in tqdm(energy_true_axis, desc='Setting up edisp normalisation', ncols=80):
        energy_true_axis_mesh, longitude_axis_true_mesh, latitude_axis_true_mesh, energy_recon_axis_mesh  = np.meshgrid(energy_val,
                                                                                                            longitudeaxistrue, 
                                                                                                            latitudeaxistrue, 
                                                                                                            energy_recon_axis,
                                                                                                            indexing='ij')
        edispvals = np.squeeze(log_edisp(energy_recon_axis_mesh.flatten(), 
                                        energy_true_axis_mesh.flatten(), 
                                        longitude_axis_true_mesh.flatten(), 
                                        latitude_axis_true_mesh.flatten(),).reshape(energy_true_axis_mesh.shape))
        edispnormvals = iterate_logspace_integration(np.squeeze(edispvals), axes=[energy_recon_axis], axisindices=[2])
        
        edispnorm.append(edispnormvals)


    psfnorm = np.squeeze(np.array(psfnorm))
    edispnorm = np.array(edispnorm)

    edispnorm[np.isneginf(edispnorm)] = 0
    psfnorm[np.isneginf(psfnorm)] = 0
    if save_results:
        np.save(save_directory+"/log_prod5_psf_normalisations.npy", psfnorm)
        np.save(save_directory+"/log_prod5_edisp_normalisations.npy", edispnorm)

    return psfnorm, edispnorm


if __name__=="__main__":
    try:
        save_directory = sys.argv[1]
    except:
        save_directory = 0

    if save_directory!=0:
        print(f"Chosen save directory is: {save_directory}")
        irf_norm_setup(save_directory=save_directory)
    else:
        print("No save directory specified.")

        irf_norm_setup()