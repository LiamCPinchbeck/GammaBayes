from typing import Literal
from astropy.io import fits
import torch
import numpy as np
from gammabayes import GammaBinning
from astropy import units as u
from gammabayes.likelihoods import IRF_LogLikelihood
from gammabayes.priors import SourceFluxDiscreteLogPrior
from gammabayes.utils import EnergySpatialTemplateInterpolator, download_and_unpack_tar, _get_package_data_directory
from gammabayes import RegularTorchInterpolator
from gammabayes.priors import StaticSourceDistTemplate, StaticDistTemplate
from pathlib import Path
import warnings




def extract_galprop_components(binning_geometry:GammaBinning, 
                                   component:Literal['pion','bremss','ics','all', 'custom']='all', 
                                   custom_galprop_fits_file_path=None,
                                   resolution:Literal['Medium', 'High']='Medium',
                                   **kwargs
                                   ):

    res_dict = {'Medium':5, 'High':6}
    file_endings_dict = {5:"j8h245c88vhzr5yu", 6:"uxwddrp6wtlkt4pd"}

    try:
        text_res = resolution
        resolution = res_dict[resolution]
        file_ending = file_endings_dict[resolution]
    except KeyError:
        raise ValueError("Invalid resolution set. Must be either 'Medium' or 'High'. There is no Low.")
    

    if custom_galprop_fits_file_path is not None:
        component = 'custom'
    


    galprop_dir_path = Path(_get_package_data_directory()/Path(f"results_54_0e03000{resolution}"))


    if component in ['pion','bremss','ics','all']:
        if not galprop_dir_path.is_dir():
            warnings.warn("""\nHey there, this looks to be the first time you are trying to use the GALPROP templates at least at the given resolution. 
Please wait up to ~10 minutes for the results to download into the package's data module.
Medium (the default) should take up about 130MB of disk space while High takes up around 4GB.\n""")
            
            download_and_unpack_tar(url=f"https://galprop.stanford.edu/wrxiv/0e03/results_54_0e03000{resolution}_{file_ending}.tar.gz", verify=False)



        if component=='all':
            return (extract_galprop_prior_template(binning_geometry=binning_geometry, irf_loglike=irf_loglike, component='pion', resolution=text_res),
                    extract_galprop_prior_template(binning_geometry=binning_geometry, irf_loglike=irf_loglike, component='bremss', resolution=text_res),
                    extract_galprop_prior_template(binning_geometry=binning_geometry, irf_loglike=irf_loglike, component='ics', resolution=text_res))
        else:

            component_dict = {'pion':'pion_decay_skymap', 'bremss':'bremss_skymap', 'ics':'ics_skymap_comp'}
            file_component = component_dict[component]

            output_file_path = _get_package_data_directory()/galprop_dir_path/Path(f"{file_component}_54_0e03000{resolution}.gz")
            

            try:
                hdu = fits.open(output_file_path)
            except FileNotFoundError:
                download_and_unpack_tar(url=f"https://galprop.stanford.edu/wrxiv/0e03/results_54_0e03000{resolution}_{file_ending}.tar.gz", verify=False)

                hdu = fits.open(output_file_path)


    elif component=='custom':
        hdu = fits.open(custom_galprop_fits_file_path)



    primary_hdu = hdu[0]
    __data = primary_hdu.data.T
    __header = primary_hdu.header


    lon_axis_1 = np.arange(__header["CRVAL1"], 
                                    __header["CRVAL1"]+__header["CDELT1"]*__header["NAXIS1"], 
                                    __header["CDELT1"])
    lat_axis_2 = np.arange(__header["CRVAL2"], 
                                    __header["CRVAL2"]+__header["CDELT2"]*__header["NAXIS2"], 
                                    __header["CDELT2"])
    energy_axis = 10**np.linspace(__header["CRVAL3"], 
                                    __header["CRVAL3"]+__header["CDELT3"]*__header["NAXIS3"], 
                                    __header["NAXIS3"])
    
    energy_axis = energy_axis/1e6

    # Checking to see if the axes are about the Galactic Centre. 
    #   If the bounds of the longitude axis multiply to something negative then they must be different signs
    #   (lower bound negative and upper bound positive)
    longitude_indices = np.arange(len(lon_axis_1))

    longitude_mask = np.append(longitude_indices[lon_axis_1>180], longitude_indices[lon_axis_1<=180])

    new_longitude_axis = lon_axis_1[lon_axis_1>180]-360
    new_longitude_axis = np.append(new_longitude_axis, lon_axis_1[lon_axis_1<=180])
    



    # Finding longitude and latitude values that fall within the bounds of the given binning geometry
    care_about_lon_mask = np.logical_and(np.where(new_longitude_axis<=binning_geometry.lon_axis[-1].numpy()+2*binning_geometry.lon_res.numpy(), True, False), 
                                         np.where(new_longitude_axis>=binning_geometry.lon_axis[0].numpy()-2*binning_geometry.lon_res.numpy(), True, False))
    care_about_lat_mask = np.logical_and(np.where(lat_axis_2<=binning_geometry.lat_axis[-1].numpy()+2*binning_geometry.lat_res.numpy(), True, False), 
                                         np.where(lat_axis_2>=binning_geometry.lat_axis[0].numpy()-2*binning_geometry.lat_res.numpy(), True, False))

    new_longitude_axis = new_longitude_axis[care_about_lon_mask]
    lat_axis_2 = lat_axis_2[care_about_lat_mask]


    # galprop_binning_geometry = GammaBinning(
    #     energy_axis=energy_axis,
    #     lon_axis=new_longitude_axis,
    #     lat_axis=lat_axis_2
    # )




    __data = __data[longitude_mask, :, :, :][care_about_lon_mask, :, :, :][:, care_about_lat_mask, :, :]



    reformatted_data_matrix = np.transpose(np.sum(__data, axis=-1), axes=(2,0,1))
    reformatted_data_matrix = reformatted_data_matrix/((1e6*energy_axis)**2)[:, None, None]*((u.TeV/u.MeV) * ((u.deg**2)/(u.sr))).to("")


    del __header
    del __data


    return {
        'energy_axis':torch.tensor(energy_axis), 
        'lon_axis':torch.tensor(new_longitude_axis), 
        'lat_axis':torch.tensor(lat_axis_2), 
        'data':torch.tensor(reformatted_data_matrix)}


def extract_and_interpolate_galprop_components(
    binning_geometry:GammaBinning, 

    component:Literal['pion','bremss','ics','all', 'custom']='all', 
    custom_galprop_fits_file_path=None,
    resolution:Literal['Medium', 'High']='Medium',
    extracted_galprop_information = None,
    **kwargs):

    if extracted_galprop_information is None:
        extracted_galprop_information = extract_galprop_components(
            binning_geometry=binning_geometry, 
            component=component, 
            custom_galprop_fits_file_path=custom_galprop_fits_file_path,
            resolution=resolution,
            **kwargs)

    template_interpolator = RegularTorchInterpolator(
        (
            np.log10(extracted_galprop_information['energy_axis']), 
            extracted_galprop_information['lon_axis'], 
            extracted_galprop_information['lat_axis']
            ), 
        extracted_galprop_information['data'])

    grid = binning_geometry.shaped_grid

    log10egrid = torch.stack([torch.log10(grid[..., 0].flatten()), grid[..., 1].flatten(), grid[..., 2].flatten()], dim=0)
    
    log_flux_interpolated = torch.log(template_interpolator(log10egrid)).reshape((grid[..., 0]).shape)

    return log_flux_interpolated


def get_galprop_static_source_flux_prior(binning_geometry:GammaBinning, 

        component:Literal['pion','bremss','ics','all', 'custom']='all', 
        custom_galprop_fits_file_path=None,
        resolution:Literal['Medium', 'High']='Medium',
        extracted_galprop_information = None,
        *args, **kwargs):

    log_flux_interpolated = extract_and_interpolate_galprop_components(
        binning_geometry=binning_geometry, 
        component=component, 
        custom_galprop_fits_file_path=custom_galprop_fits_file_path,
        resolution=resolution,
        extracted_galprop_information = extracted_galprop_information)

    return StaticSourceDistTemplate(
                source_flux_tensor_logtemplate = log_flux_interpolated,
                binning_geometry=binning_geometry,
                *args, **kwargs
                )

    