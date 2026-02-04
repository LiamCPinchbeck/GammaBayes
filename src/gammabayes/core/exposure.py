# Will contain a class for handling effective area, 
#   observation time and multi-observation bin overlaps
#   i.e. Source Flux Exposures

import numpy as np
from .binning_geometry import GammaBinning
from .reg_interpolator import RegularTorchInterpolator
from astropy import units as u

import copy
from warnings import warn
import torch


def trivial_log_aeff(energy, lon, lat, pointing_dir=None):
    return energy*0.

class GammaLogExposure:
    def __init__(self, 
                 binning_geometry:GammaBinning, 
                 irfs=None, 
                 log_exposure_map = None,
                 pointing_dirs = None,

                 # One second is taken as most differential fluxes are per second, so this works as a unit where the absolute values are the quantities don't change
                 live_times=1*u.s, 
                 
                 use_log_aeff: bool = True,
                 apply_bin_widths: bool = True,
                 # By default the units of the effective area are taken to be u.cm**2 and intermediary calculations use seconds when needed
                 unit_bases = None, 
                 log_obs_time_map = None
                 ):
        if unit_bases is None:
            unit_bases = [u.cm, u.s] 
        
        self.use_log_aeff = use_log_aeff
        self.apply_bin_widths = apply_bin_widths

        if isinstance(log_exposure_map, GammaLogExposure):
            for attr, value in log_exposure_map.__dict__.items():
                setattr(self, attr, copy.deepcopy(value))
        else:
            self.binning_geometry = binning_geometry

            self.irfs = irfs

            if self.use_log_aeff:
                self.log_aeff = self.irfs.log_aeff
            else:
                self.log_aeff = trivial_log_aeff

            self.aeff_units = self.irfs.aeff_units

            self.__parse_livetime(live_times=live_times)
            self.__parse_pointing_dirs(pointing_dirs=pointing_dirs)
                
            self.unit = self.live_time_units*self.aeff_units

            # Gets rid of annoying things like u.hr/u.s not being simplified
            try:
                decomposed_unit = self.unit.decompose(unit_bases)
                self.unit = ((1*decomposed_unit).decompose(unit_bases)).unit
                self.log_unit_converter = torch.log(((1*decomposed_unit).decompose(unit_bases)).value)
            except:
                self.unit = self.unit
                self.log_unit_converter = 0.
                


            self._reset_cache()

            if not(log_exposure_map is None):
                self.log_exposure_map = log_exposure_map
                self.log_obs_time_map = log_obs_time_map

                if self.log_obs_time_map is None:
                    self.log_obs_time_map = (live_times/u.s).to("")

                if self.log_obs_time_map.dim() == 0:
                    self.log_obs_time_map = torch.full(self.binning_geometry.shape, self.log_obs_time_map)

                self._exp_interpolator = RegularTorchInterpolator(self.binning_geometry.axes, torch.exp(self.log_exposure_map))
                self._exp_obs_time_interpolator = RegularTorchInterpolator(self.binning_geometry.axes, torch.exp(self.log_obs_time_map))
            else:
                self.refresh()

                

    def __parse_livetime(self, live_times):


        if isinstance(live_times, u.Quantity):
            if live_times.isscalar:
                live_times = [live_times]
            else:
                live_times = live_times
        elif isinstance(live_times, (int, float)):
            live_times = [live_times]
        elif isinstance(live_times, (np.ndarray, list, tuple, torch.Tensor)):
            live_times = live_times
        else:
            warn("Cannot interpret livetime input. Must be list of scalars or scalar. Not sure how you gave something else.")
        
        self.live_times = live_times


        try:
            self.live_time_units = self.live_times[0].unit
        except:
            self.live_time_units = u.s

        try:
            self.live_times_values = [live_time.to(self.live_time_units).value for live_time in self.live_times]
        except:
            self.live_times_values = [live_time for live_time in self.live_times]


        self.live_times = torch.tensor(self.live_times_values)*(self.live_time_units/u.s).to("")



    def __parse_pointing_dirs(self, pointing_dirs):

        if np.asarray(pointing_dirs).ndim <2:
            pointing_dirs = torch.stack([pointing_dirs], dim=0)

        self.pointing_dirs = pointing_dirs

        if hasattr(self.irfs, "pointing_dir") and (self.pointing_dirs is None):
            self.pointing_dirs = torch.stack([self.irfs.pointing_dir], dim=0)

        elif self.pointing_dirs is None:
            self.pointing_dirs = torch.stack([self.binning_geometry.spatial_centre], dim=0)


        self.pointing_dirs = torch.tensor(self.pointing_dirs)



    def __call__(self, *args, **kwargs):
        return torch.log(self.exp_interpolator(*args, **kwargs))+self.log_unit_converter


    # Support for indexing like a list or array
    def __getitem__(self, key: int | slice):

        return self.log_exposure_map[key]

    def __add__(self, other):

        if isinstance(other, GammaLogExposure):

            if self.binning_geometry!=other.binning_geometry:
                raise NotImplemented("Adding exposures for different binning geometries is currently not supported.")


            other_unit_scaling = (self.aeff_units/other.aeff_units).to("")

            new_exposure_map = torch.logaddexp(self.log_exposure_map, torch.log(other_unit_scaling)+other.log_exposure_map)


            return GammaLogExposure(binning_geometry=self.binning_geometry, 
                                    log_exposure_map=new_exposure_map,
                                    irfs=self.irfs,
                                    pointing_dirs= self.pointing_dirs.extend(other.pointing_dirs),
                                    lives_times=self.live_times.extend(other.live_times),
                                    )
        else:

            return other+self.log_exposure_map
        


    def _same_as_cached(self, pointing_dirs=None, live_times=None):


        if pointing_dirs is None:
            same_as_cached = True
            return same_as_cached
        
        if np.asarray(pointing_dirs).ndim <2:

            same_as_cached = torch.any((torch.tensor(self.pointing_dirs) == pointing_dirs).all(axis=1))

            return same_as_cached

        same_as_cached = torch.array_equiv(torch.sort(pointing_dirs, axis=0), torch.sort(self.pointing_dirs, axis=0))


        return same_as_cached
    

    def _reset_cache(self):


        if self.pointing_dirs is None:
            raise ValueError()

        self._cached_pointing_dirs = self.pointing_dirs
        self._cached_live_times = self.live_times



    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
        

    def refresh(self):

        log_exposure_vals = -torch.tensor(torch.inf)

        for pointing_dir, live_time in zip(self.pointing_dirs, self.live_times):
            print(live_time)
            log_exposure_vals = torch.logaddexp(log_exposure_vals, self.log_aeff(*self.binning_geometry.axes_mesh, pointing_dir=pointing_dir)+torch.log(live_time)+self.log_unit_converter)

    
        if self.apply_bin_widths:
            self.log_exposure_map = log_exposure_vals + torch.log(self.binning_geometry.bin_width_mat)
        else:
            self.log_exposure_map = log_exposure_vals

        # Have to interpolate exposure not log_exposure due to possible -inf values
        self._exp_interpolator = RegularTorchInterpolator(self.binning_geometry.axes, torch.exp(self.log_exposure_map))

        self._reset_cache()
        
        return self.log_exposure_map
    

    def add_single_exposure(self, pointing_dir, live_time):


        self.pointing_dirs = self.pointing_dirs.append(pointing_dir)
        self.live_times = self.live_times.append(live_time)

        if hasattr(live_time, "unit"):
            live_time = live_time.to(self.live_time_units)
        else:
            live_time = live_time*self.live_time_units



        self.log_exposure_map = torch.logaddexp(self.log_exposure_map, self.log_aeff(*self.binning_geometry.axes_mesh, pointing_dir=pointing_dir)+torch.log(live_time.value)+self.log_unit_converter)



        # Have to interpolate exposure not log_exposure due to possible -inf values
        self._exp_interpolator = RegularTorchInterpolator(self.binning_geometry.axes, torch.exp(self.log_exposure_map))

        self._reset_cache()
        
        return self.log_exposure_map
            
    
    def exp_interpolator(self, energy, lon, lat, *args, pointing_dirs=None, live_times=None, **kwargs):

        if not self._same_as_cached(pointing_dirs=pointing_dirs):
            if torch.array(pointing_dirs).size>2:
                self.pointing_dirs = pointing_dirs
                self.live_times = live_times

                self.refresh()

            else:
                self.add_single_exposure(pointing_dir=pointing_dirs, live_times=live_times)

        return self._exp_interpolator((energy, lon , lat), *args, **kwargs)
        

    def peek(self, fig_kwargs=None, pcolormesh_kwargs=None, plot_kwargs=None, **kwargs):

        if fig_kwargs is None:
            fig_kwargs = {}

        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {}

        if plot_kwargs is None:
            plot_kwargs = {}

        from matplotlib import pyplot as plt
        from matplotlib import patches
        from matplotlib.colors import LogNorm
        from gammabayes.utils.integration import iterate_logspace_integration

        fig_kwargs.update(kwargs)

        if 'figsize' not in fig_kwargs:
            fig_kwargs['figsize'] = (12, 6)

        integrated_energy_exposure = torch.logsumexp(
            self.log_exposure_map, 
            dim=0).detach().cpu().numpy()
        

        integrated_spatial_exposure = torch.logsumexp(torch.logsumexp(self.log_exposure_map, dim=1), dim=1).detach().cpu().numpy()
        

        integrated_lat_exposure = torch.logsumexp(
            self.log_exposure_map+
            torch.log(self.binning_geometry.lat_bin_widths
                )[None, None, :], dim=2).detach().cpu().numpy()


        log_exposure_map_to_slice = self.log_exposure_map.clone().detach().cpu().numpy()

        weighted_mean_pointing_dir = torch.sum(torch.tensor(self.pointing_dirs).T*self.live_times, axis=1)/torch.sum(self.live_times)

        try:
            weighted_mean_pointing_dir = [weighted_mean_pointing_dir[0], weighted_mean_pointing_dir[1]]
        except:
            weighted_mean_pointing_dir = weighted_mean_pointing_dir

        energy_slice = torch.abs(self.binning_geometry.energy_axis-1).argmin().detach().cpu().numpy()
        lon_slice = torch.abs(self.binning_geometry.lon_axis-weighted_mean_pointing_dir[0]).argmin().detach().cpu().numpy()
        lat_slice = torch.abs(self.binning_geometry.lat_axis-weighted_mean_pointing_dir[1]).argmin().detach().cpu().numpy()

        energy_slice_val = self.binning_geometry.energy_axis[energy_slice].detach().cpu().numpy()
        lon_slice_val = self.binning_geometry.lon_axis[lon_slice].detach().cpu().numpy()
        lat_slice_val = self.binning_geometry.lon_axis[lat_slice].detach().cpu().numpy()

        plotting_energy_axis = self.binning_geometry.energy_axis.clone().detach().cpu().numpy()
        plotting_lon_axis = self.binning_geometry.lon_axis.clone().detach().cpu().numpy()
        plotting_lat_axis = self.binning_geometry.lat_axis.clone().detach().cpu().numpy()



        slice_coord_str = f"({lon_slice_val:.2g}, {lat_slice_val:.2g})"
        
        if 'norm' not in pcolormesh_kwargs:
            pcolormesh_kwargs['norm'] = 'log'

        fig, ax = plt.subplots(2, 3, **fig_kwargs)


        ax[0,0].plot(plotting_energy_axis, np.exp(log_exposure_map_to_slice[:, lon_slice, lat_slice].T), label=f'Exposure at {slice_coord_str}', **plot_kwargs)
        ax[0,0].set_xscale('log')
        ax[0,0].set_xlabel(r"Energy [TeV]")
        ax[0,0].set_ylabel(r"Exposure ["+(self.unit).to_string('latex')+"]",)
        ax[0,0].legend()
        ax[0,0].grid(which='major', c='grey', ls='--', alpha=0.4)


        ax[1,0].plot(plotting_energy_axis, np.exp(integrated_spatial_exposure.T), **plot_kwargs)
        ax[1,0].set_xscale('log')
        ax[1,0].set_xlabel(r"Energy [TeV]")
        ax[1,0].set_ylabel(r"Integrated Exposure ["+(self.unit).to_string('latex')+"*deg^2]",)

        pcm = ax[0,1].pcolormesh(plotting_lon_axis, plotting_lat_axis, 
                                 np.exp(log_exposure_map_to_slice[energy_slice, :, :].T),
                                 **pcolormesh_kwargs)
        ax[0,1].legend(title="Slice at 1 TeV")
        plt.colorbar(mappable=pcm, label=r"Exposure ["+(self.unit).to_string('latex')+"]", ax= ax[0,1])
        ax[0,1].set_xlabel(r"Longitude [deg]")
        ax[0,1].set_ylabel(r"Latitude [deg]")
        ax[0,1].set_aspect('equal', adjustable='box')
        ax[0,1].invert_xaxis()


        int_pcm = ax[1,1].pcolormesh(plotting_lon_axis, plotting_lat_axis, np.exp(integrated_energy_exposure.T), **pcolormesh_kwargs)
        plt.colorbar(mappable=int_pcm, label=r"Integrated Exposure [TeV]", ax= ax[1,1])
        ax[1,1].set_xlabel(r"Longitude [deg]")
        ax[1,1].set_ylabel(r"Latitude [deg]")
        ax[1,1].set_aspect('equal', adjustable='box')
        ax[1,1].invert_xaxis()



        pcm = ax[0,2].pcolormesh(plotting_lon_axis, plotting_energy_axis, 
                                 np.exp(log_exposure_map_to_slice[:, :, lat_slice]),
                                 **pcolormesh_kwargs)
        ax[0,2].legend(title=f"Slice at lat={lat_slice_val:.2g} deg")
        plt.colorbar(mappable=pcm, label=r"Exposure ["+(self.unit).to_string('latex')+"]", ax= ax[0,2])
        ax[0,2].set_xlabel(r"Longitude [deg]")
        ax[0,2].invert_xaxis()
        ax[0,2].set_ylabel(r"Energy [TeV]")
        ax[0,2].set_yscale('log')

        int_pcm = ax[1,2].pcolormesh(plotting_lon_axis, plotting_energy_axis, np.exp(integrated_lat_exposure), **pcolormesh_kwargs)
        plt.colorbar(mappable=int_pcm, label=r"Integrated Exposure ["+(self.unit).to_string('latex')+"*deg]", ax= ax[1,2])
        ax[1,2].set_xlabel(r"Longitude [deg]")
        ax[1,2].invert_xaxis()
        ax[1,2].set_yscale('log')
        ax[1,2].set_ylabel(r"Energy [TeV]")

        plt.tight_layout()

        return fig, ax

