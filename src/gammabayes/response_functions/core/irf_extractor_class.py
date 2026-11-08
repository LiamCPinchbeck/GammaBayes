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


        energy_temp = energy.detach().cpu().numpy()
        offsets = haversine(lon, lat, pointing_dir[0], pointing_dir[1]).detach().cpu().numpy()*u.deg

        result = torch.tensor(self.aeff_default.evaluate(energy_true = energy_temp*u.TeV, 
                                offset=offsets).to(self.aeff_units).value).log()

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

        offset  = haversine(lon, lat, pointing_dir[0], pointing_dir[1]).detach().cpu().numpy()
        energy_temp = energy.detach().cpu().numpy()




        return torch.tensor(self.CCR_BKG.evaluate(energy=energy_temp*u.TeV, offset=offset*u.deg).to(self.CCR_BKG_units).value).log()
    

    
    # ------------------------------------------------------------------
    # Energy-calibrated PSF (A3)
    # ------------------------------------------------------------------
    def psf_containment_radius(self, energy_true, offset, fraction=0.68, rad_max_deg=3.0, n_rad=600):
        """Containment radius [deg] of the PSF at (energy_true [TeV], offset [deg]).

        Computed by integrating ``psf.evaluate`` on a log-spaced radial grid
        and inverting the cumulative, so it does not depend on the gammapy
        version's ``containment_radius`` signature.  ``energy_true`` and
        ``offset`` are broadcast against each other; returns a numpy array.
        """
        energy_true = np.atleast_1d(np.asarray(energy_true, dtype=float))
        offset = np.atleast_1d(np.asarray(offset, dtype=float))
        E, O = np.broadcast_arrays(energy_true[:, None], offset[None, :])
        rad = np.logspace(-3, np.log10(rad_max_deg), n_rad)              # deg
        # psf(E, offset, rad) in 1/sr -> integrate 2 pi r dr (sr)
        vals = self.psf_default.evaluate(energy_true=E[..., None] * u.TeV,
                                         rad=rad[None, None, :] * u.deg,
                                         offset=O[..., None] * u.deg).to(self.psf_units).value
        rad_sr = np.deg2rad(rad)
        dcum = 0.5 * (vals[..., 1:] + vals[..., :-1]) * 2 * np.pi * 0.5 * (rad_sr[1:] + rad_sr[:-1]) * np.diff(rad_sr)
        cum = np.concatenate([np.zeros(E.shape + (1,)), np.cumsum(dcum, axis=-1)], axis=-1)
        cum = cum / np.maximum(cum[..., -1:], 1e-300)
        out = np.empty(E.shape)
        for idx in np.ndindex(E.shape):
            out[idx] = np.interp(fraction, cum[idx], rad)
        return out

    def psf_sigma_vs_energy(self, binning_geometry, fraction=0.68, weighting="aeff",
                            counts=None, pointing_dir=None, n_offset_nodes=16, pixel_floor=True):
        """FoV-averaged Gaussian-equivalent PSF width sigma(E_true) [deg] on the
        geometry's true-energy axis.

        sigma = R_f / sqrt(-2 ln(1 - f))   (R_68 / 1.5096 for f = 0.68),
        i.e. the sigma of the 2-D Gaussian with the same containment radius.

        weighting: "uniform" -- plain pixel average over the FoV,
                   "aeff"    -- weighted by A_eff(E, offset) (default),
                   "counts"  -- weighted by a user-supplied (E, lon, lat) cube
                                (e.g. the Asimov background) via ``counts``.
        pixel_floor: add the pixel variance (Delta^2/12 per axis) in quadrature
                     so sub-pixel PSFs never collapse to a delta kernel.
        Returns a torch tensor (E,) in the geometry's dtype.
        """
        geom = binning_geometry
        if pointing_dir is None:
            pointing_dir = torch.as_tensor(self.pointing_dir if hasattr(self, "pointing_dir") else [0., 0.],
                                           dtype=geom.lon_axis.dtype)
        lon, lat = torch.meshgrid(geom.lon_axis, geom.lat_axis, indexing="ij")
        off = haversine(lon.flatten(), lat.flatten(), pointing_dir[0], pointing_dir[1]).detach().cpu().numpy()
        E = geom.energy_axis.detach().cpu().numpy()

        # R_f on a few offset nodes, then interpolate to every pixel
        nodes = np.linspace(0., max(off.max(), 1e-3), n_offset_nodes)
        R_nodes = self.psf_containment_radius(E, nodes, fraction=fraction)          # (E, n_nodes)
        R_pix = np.stack([np.interp(off, nodes, R_nodes[e]) for e in range(len(E))])  # (E, S)

        if weighting == "uniform":
            w = np.ones_like(R_pix)
        elif weighting == "aeff":
            w = self.aeff_default.evaluate(energy_true=E[:, None] * u.TeV, offset=off[None, :] * u.deg).to(self.aeff_units).value
            w = np.nan_to_num(w, nan=0.0)
        elif weighting == "counts":
            if counts is None:
                raise ValueError("weighting='counts' needs the counts cube")
            w = torch.as_tensor(counts).detach().cpu().numpy().reshape(len(E), -1)
        else:
            raise ValueError("weighting must be 'uniform', 'aeff' or 'counts'")
        w = np.where(w > 0, w, 0.0)
        wsum = w.sum(1)
        R = np.where(wsum > 0, (w * R_pix).sum(1) / np.maximum(wsum, 1e-300), R_pix.mean(1))
        sigma = R / np.sqrt(-2.0 * np.log(1.0 - fraction))
        if pixel_floor:
            dlon = float(geom.lon_axis[1] - geom.lon_axis[0])
            dlat = float(geom.lat_axis[1] - geom.lat_axis[0])
            sigma = np.sqrt(sigma ** 2 + (dlon ** 2 + dlat ** 2) / 24.0)   # mean of the two per-axis Delta^2/12
        return torch.as_tensor(sigma, dtype=geom.energy_axis.dtype)

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





        
