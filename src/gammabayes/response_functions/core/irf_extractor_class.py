from gammabayes import haversine, resources_dir
from astropy import units as u
from astropy.units import Quantity
from gammapy.irf import load_irf_dict_from_file
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
import os, zipfile, time, numpy as np, importlib.resources as pkg_resources

from gammabayes.response_functions.instrument_specific_interfaces.CTAO_IRFs.CTAO_irf_file_utils import find_ctao_irf_file_path
from gammabayes.response_functions.instrument_specific_interfaces.HESS_IRFs import extract_and_generate_hess_data
from gammapy.data import DataStore


import torch


class IRFExtractor(object):
    def __init__(self, 
                 zenith_angle:int=20, 
                 hemisphere:str='South', 
                 prod_vers=5, 
                 observation_time: float|u.Quantity = 50*u.hr,
                 file_path: str=None,
                 psf_units: u.Unit = (1/u.deg**2).unit,
                 edisp_units: u.Unit = (1/u.TeV).unit,
                 aeff_units: u.Unit = u.cm**2,
                 CCR_BKG_units: u.Unit = (1/(u.deg**2*u.TeV*u.s)).unit,
                 instrument: str ='CTAO',
                 obs_id: int = None,
                 pointing_dir= torch.tensor([0, 0])):

        self._format_instrument(instrument)


        self.file_path = file_path
        self.observation_time = observation_time
        if hasattr(self.observation_time, "unit"):
            irf_time_in_seconds = self.observation_time.to(u.s)


        else:
            irf_time_in_seconds = 180000

        if self.file_path is None:
            # self.file_path = resources_dir+f'/irf_fits_files/Prod5-South-20deg-AverageAz-14MSTs37SSTs.{irf_time_in_seconds}s-v0.1.fits'

            if self.instrument in ['CTA', 'CTAO']:
                self._CTAO_init(zenith_angle=zenith_angle, hemisphere=hemisphere, prod_vers=prod_vers)
            else:
                self._HESS_init(pointing_dir=pointing_dir, obs_id=obs_id)


        else:
            self.extracted_default_irfs  = load_irf_dict_from_file(self.file_path)


        self.psf_units = psf_units
        self.edisp_units = edisp_units
        self.aeff_units = aeff_units
        self.CCR_BKG_units = CCR_BKG_units


        self.edisp_default      = self.extracted_default_irfs['edisp']
        # self.edisp_default.normalize()

        self.psf_default        = self.extracted_default_irfs['psf']


        if hasattr(self.psf_default, 'to_psf3d'):
            self.psf3d              = self.psf_default.to_psf3d()
        else:
            self.psf3d              = self.psf_default

        # self.psf3d.normalize()

        self.aeff_default       = self.extracted_default_irfs['aeff']
        self.CCR_BKG            = self.extracted_default_irfs['bkg'].to_2d()

        self.zenith = zenith_angle
        self.hemisphere = hemisphere
        self.prod_vers = prod_vers


    def _format_instrument(self, instrument_str:str):
        
        if instrument_str in ['CTA', 'CTAO']:
            self.instrument = 'CTAO'
        elif instrument_str == 'HESS':
            self.instrument = instrument_str
        else:
            raise ValueError("You have specified an instrument that GammaBayes currently does not support. Please select either CTAO or HESS")





    def _CTAO_init(self, zenith_angle, hemisphere, prod_vers, *args, **kwargs):
        prod_version, fits_file_path = find_ctao_irf_file_path(zenith_angle=zenith_angle, 
                                            hemisphere=hemisphere, 
                                            prod_vers=prod_vers)
        
        self.extracted_default_irfs  = load_irf_dict_from_file(fits_file_path)



    def log_aeff(self, energy, lon, lat, pointing_dir=torch.tensor([0,0]), parameters={}):

        result = torch.tensor(self.aeff_default.evaluate(energy_true = energy*u.TeV, 
                                offset=haversine(
                                    lon, lat, pointing_dir[0], pointing_dir[1])*u.deg).to(self.aeff_units).value).log()

        result = torch.where(torch.isneginf(result), -100, result)

        return result


    def log_edisp(self, recon_energy, 
                  true_energy, true_lon, true_lat, 
                  pointing_dir=torch.tensor([0.,0.]), parameters:dict={}, migration_cut=torch.tensor(11.)):
        """
        Wrapper for the Gammapy interpretation of the CTA energy dispersion function.

        Args:
            recon_energy (torch.Tensor): Measured energy value by the CTA.
            true_energy (torch.Tensor): True energy of a gamma-ray event detected by the CTA.
            true_lon (torch.Tensor): True FOV longitude of a gamma-ray event detected by the CTA.
            true_lat (torch.Tensor): True FOV latitude of a gamma-ray event detected by the CTA.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: Natural log of the CTA energy dispersion likelihood for the given gamma-ray event data.
        """


        offset = haversine(true_lon, true_lat, pointing_dir[0], pointing_dir[1])

        migration = recon_energy/true_energy



        edisp_val = torch.where(torch.logical_and(migration<migration_cut, migration>1/migration_cut), torch.tensor((self.edisp_default.evaluate(energy_true=true_energy*u.TeV,
                                                        migra = migration, 
                                                        offset=offset*u.deg)/(recon_energy*u.TeV)).to(self.edisp_units).value), 0)


        # edisp output is dimensionless when it should have units of 1/TeV
        log_output = (edisp_val).log()

        return log_output



    def log_psf(self, recon_lon, recon_lat, 
                true_energy, true_lon, true_lat, 
                pointing_dir=torch.tensor([0.,0.]), parameters:dict={}):
        """
        Wrapper for the Gammapy interpretation of the CTA point spread function.

        Args:
            recon_lon (torch.Tensor): Measured FOV longitude of a gamma-ray event detected by the CTA.
            recon_lat (torch.Tensor): Measured FOV latitude of a gamma-ray event detected by the CTA.
            true_energy (torch.Tensor): True energy of a gamma-ray event detected by the CTA.
            true_lon (torch.Tensor): True FOV longitude of a gamma-ray event detected by the CTA.
            true_lat (torch.Tensor): True FOV latitude of a gamma-ray event detected by the CTA.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: Natural log of the CTA point spread function likelihood for the given gamma-ray event data.
        """

        rad = haversine(recon_lon.flatten(), recon_lat.flatten(), true_lon.flatten(), true_lat.flatten(),).flatten()

        offset  = haversine(true_lon.flatten(), true_lat.flatten(), pointing_dir[0], pointing_dir[1]).flatten()

        output = torch.tensor(self.psf_default.evaluate(energy_true=true_energy*u.TeV, rad = rad, 
                                                  offset=offset*u.deg).to(self.psf_units).value).log()
                
        return output



    
    def log_bkg_CCR(self, energy, lon, lat, 
                    spectral_parameters:dict={}, spatial_parameters:dict={},
                    pointing_dir=torch.tensor([0.,0.]), ):
        """
        Wrapper for the Gammapy interpretation of the log of the CTA's background charged cosmic-ray mis-identification rate.

        Args:
            energy (torch.Tensor): True energy of a gamma-ray event detected by the CTA.
            lon (torch.Tensor): True FOV longitude of a gamma-ray event detected by the CTA.
            lat (torch.Tensor): True FOV latitude of a gamma-ray event detected by the CTA.
            spectral_parameters (dict, optional): Spectral parameters. Defaults to {}.
            spatial_parameters (dict, optional): Spatial parameters. Defaults to {}.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: Natural log of the charged cosmic ray mis-identification rate for the CTA.
        """

        offset  = haversine(lon, lat, pointing_dir[0], pointing_dir[1])



        return torch.tensor(self.CCR_BKG.evaluate(energy=energy*u.TeV, offset=offset*u.deg).to(self.CCR_BKG_units).value).log()
    

    
    def plot_edisp_migration(self, meas_binning_geom, true_binning_geom, cmap='Blues', *args, **kwargs):
        from matplotlib import pyplot as plt

        centre_spatial = meas_binning_geom.spatial_centre

        meas_energy_axis = meas_binning_geom.energy_axis
        true_energy_axis = true_binning_geom.energy_axis

        energy_meshes = torch.meshgrid(true_energy_axis, meas_energy_axis, indexing='ij')
        flattened_energy_meshes = [mesh.flatten() for mesh in energy_meshes]


        edisp_values = self.log_edisp(
            recon_energy=flattened_energy_meshes[0], 
            true_energy=flattened_energy_meshes[1],
            true_lon=centre_spatial[0],
            true_lat=centre_spatial[1], 
            migration_cut=torch.tensor(11.)).reshape(energy_meshes[0].shape).exp()

        fig, ax = plt.subplots(1,1)
        pcm=ax.pcolormesh(meas_energy_axis, true_energy_axis, edisp_values.T, norm='log', cmap=cmap)
        plt.colorbar(pcm, ax=ax, label=r'$\text{E}_{\text{disp}} \text{[1/TeV]}$')
        ax.set(
            xscale='log',
            yscale='log',
            xlabel=r'$\text{E}_{\text{m}} \text{[TeV]}$',
            ylabel=r'$\text{E}_{\text{t}} \text{[TeV]}$',
        )
        return fig, ax

    def plot_edisp_slices(self, meas_binning_geom, true_binning_geom, num_true_vals=5, cmap='Blues', *args, **kwargs):
        from matplotlib import pyplot as plt
        from matplotlib.pyplot import get_cmap

        cmap = get_cmap(cmap)

        centre_spatial = meas_binning_geom.spatial_centre

        meas_energy_axis = meas_binning_geom.energy_axis
        true_energy_axis_size = len(true_binning_geom.energy_axis[2:-2])

        true_energy_spacing = int(true_energy_axis_size/num_true_vals)
        true_energy_axis = true_binning_geom.energy_axis[2:-2:true_energy_spacing]
        num_true_vals = len(true_energy_axis)

        energy_meshes = torch.meshgrid(true_energy_axis, meas_energy_axis, indexing='ij')
        flattened_energy_meshes = [mesh.flatten() for mesh in energy_meshes]


        edisp_values = self.log_edisp(
            recon_energy=flattened_energy_meshes[0], 
            true_energy=flattened_energy_meshes[1],
            true_lon=centre_spatial[0],
            true_lat=centre_spatial[1], 
            migration_cut=torch.tensor(10.)).reshape(energy_meshes[0].shape).exp()

        fig, ax = plt.subplots(1,1)
        for true_energy_idx, true_energy_val in enumerate(true_energy_axis):
            ax.loglog(meas_energy_axis, edisp_values[true_energy_idx, :].T, c=cmap((true_energy_idx/num_true_vals+0.4)/1.4))

        ax.set(
            ylabel=r'$\text{E}_{\text{disp}} \text{[1/TeV]}$',
            xlabel=r'$\text{E}_{\text{m}} \text{[TeV]}$'
            )
        return fig, ax

    

    def plot_edisp_slices_in_pos(self, meas_binning_geom, true_binning_geom, offsets=torch.tensor([0,1, 2, 3]), true_energy=None, cmap='Blues', *args, **kwargs):
        from matplotlib import pyplot as plt
        from matplotlib.pyplot import get_cmap

        cmap = get_cmap(cmap)

        centre_spatial = meas_binning_geom.spatial_centre

        meas_energy_axis = meas_binning_geom.energy_axis

        size_true_binning_energy_axis = len(true_binning_geom.energy_axis)
        if true_energy is None:
            centre_energy = true_binning_geom.energy_axis[int(size_true_binning_energy_axis/2)]
        else:
            centre_energy = true_energy

        
        energy_meshes = torch.meshgrid(centre_energy, meas_energy_axis, indexing='ij')
        flattened_energy_meshes = [mesh.flatten() for mesh in energy_meshes]


        fig, ax = plt.subplots(1,1)

        num_offsets = len(offsets)
        for pos_idx, offset in enumerate(offsets):
            edisp_values = self.log_edisp(
                recon_energy=flattened_energy_meshes[0], 
                true_energy=flattened_energy_meshes[1],
                true_lon=centre_spatial[0]+offset,
                true_lat=centre_spatial[1], 
                migration_cut=torch.tensor(10.)).reshape(energy_meshes[0].shape).exp()

            ax.loglog(meas_energy_axis, edisp_values.squeeze(), c=cmap((pos_idx/num_offsets+0.5)/1.5))

        ax.set(
            ylabel=r'$\text{E}_{\text{disp}} \text{[1/TeV]}$',
            xlabel=r'$\text{E}_{\text{m}} \text{[TeV]}$'
            )
        return fig, ax





        
