from typing import Literal
from astropy.io import fits
import numpy as np
from gammabayes import GammaBinning
from astropy import units as u
from gammabayes.likelihoods import IRF_LogLikelihood
from gammabayes.priors import SourceFluxDiscreteLogPrior
from gammabayes.utils import EnergySpatialTemplateInterpolator, download_and_unpack_tar, _get_package_data_directory
from pathlib import Path
import warnings




def extract_galprop_prior_template(true_binning_geometry:GammaBinning, 
                                   irf_loglike:IRF_LogLikelihood, 
                                   component:Literal['pion','bremss','ics','all', 'custom']='all', 
                                   custom_galprop_fits_file_path=None,
                                   resolution:Literal['Medium', 'High']='Medium',
                                   log_exposure_map = None,
                                   pointing_dirs=None,
                                   live_times=None,
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
            return (extract_galprop_prior_template(true_binning_geometry=true_binning_geometry, irf_loglike=irf_loglike, component='pion', resolution=text_res),
                    extract_galprop_prior_template(true_binning_geometry=true_binning_geometry, irf_loglike=irf_loglike, component='bremss', resolution=text_res),
                    extract_galprop_prior_template(true_binning_geometry=true_binning_geometry, irf_loglike=irf_loglike, component='ics', resolution=text_res))
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
    


    # Checking to see if the axes are about the Galactic Centre. 
    #   If the bounds of the longitude axis multiply to something negative then they must be different signs
    #   (lower bound negative and upper bound positive)
    longitude_indices = np.arange(len(lon_axis_1))

    longitude_mask = np.append(longitude_indices[lon_axis_1>180], longitude_indices[lon_axis_1<=180])

    new_longitude_axis = lon_axis_1[lon_axis_1>180]-360
    new_longitude_axis = np.append(new_longitude_axis, lon_axis_1[lon_axis_1<=180])
    



    # Finding longitude and latitude values that fall within the bounds of the given binning geometry
    care_about_lon_mask = np.logical_and(np.where(new_longitude_axis<=true_binning_geometry.lon_axis[-1].value+2*true_binning_geometry.lon_res.value, True, False), 
                                         np.where(new_longitude_axis>=true_binning_geometry.lon_axis[0].value-2*true_binning_geometry.lon_res.value, True, False))
    care_about_lat_mask = np.logical_and(np.where(lat_axis_2<=true_binning_geometry.lat_axis[-1].value+2*true_binning_geometry.lat_res.value, True, False), 
                                         np.where(lat_axis_2>=true_binning_geometry.lat_axis[0].value-2*true_binning_geometry.lat_res.value, True, False))





    galprop_binning_geometry = GammaBinning(
        energy_axis=energy_axis*u.MeV,
        lon_axis=new_longitude_axis[care_about_lon_mask]*u.deg,
        lat_axis=lat_axis_2[care_about_lat_mask]*u.deg
    )




    __data = __data[longitude_mask, :, :, :][care_about_lon_mask, :, :, :][:, care_about_lat_mask, :, :]



    reformatted_data_matrix = np.transpose(np.sum(__data, axis=-1), axes=(2,0,1))



    del __header
    del __data

    template_model = EnergySpatialTemplateInterpolator(
        binning_geometry=galprop_binning_geometry,
        data=reformatted_data_matrix/galprop_binning_geometry.energy_axis.value[:, None, None]**2*((u.TeV/u.MeV) * ((u.m)**2/(u.cm)**2) * ((u.deg**2)/(u.sr))).to(""),
        interpolation_method='linear'
        )

    # Subsequent units for all components is originally MeV^2 cm^{-2} s^{-1} sr^{-1} MeV^{-1}
        # After above manipulation is m^{-2} s^{-1} deg^{-2} TeV^{-1}



    template_prior = SourceFluxDiscreteLogPrior(
        name=component+'_template_prior',
        binning_geometry=true_binning_geometry,
        irf_loglike=irf_loglike,
        log_flux_function=template_model,
        log_exposure_map=log_exposure_map,
        pointing_dirs=pointing_dirs,
        live_times=live_times,
        **kwargs
    )



    return template_prior