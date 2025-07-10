
from gammabayes.utils.integration import logspace_simpson, iterate_logspace_integration
from gammabayes import GammaBinning, GammaObs
from gammabayes.priors import DiscreteLogPrior
from gammabayes.hyper_inference import MTree
from .inverse_transform_sampling import integral_inverse_transform_sampler
import numpy as np
from tqdm import tqdm
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy import special
import time


# Intention with script
#   - Current default behaviour when sampling discrete log priors is to 
#           1. define a binning geometry
#           2. sample the prior according to this geometry to construct a discrete prior density matrix (with jacobian added)
#           3. flatten this matrix
#           4. Create a discrete cdf (requiring cdf[-1] = 1)
#           5. Create random samples from 0 to 1
#           6. Find what indices of the discrete cdf match these best
#           7. Samples of the discrete log prior transform are then these indices which slice the relevant measurement axes
#   - The main issue with is the effective integration scheme is a Riemann sum. Due to various things being very localised in space
#       or the fact that most of the spectral distributions are power-law like this is very limiting
#   - I don't want to change the default sampling scheme because it is already pretty computationally intensive as it is
#   - But I might need the ability to do really high resolution/accurate maps that require integration _within the bins_ that I've defined

# Reasons for using a class
#   - Will need a fair number of inputs including maybe 4 sets of binning geometries and angular integration scheme arguments
#       -- Hence a fair bit of input cleaning/fills will be required
#   - I can always make it just a series of functions later on


class IntraBinSampling:
    def __init__(self, priors:list[DiscreteLogPrior], 
                 default_true_binning_geometry:GammaBinning, default_recon_binning_geometry:GammaBinning,
                 logenergy_true_subdivisions=9, longitude_true_subdivisions=9, latitude_true_subdivisions=9,
                 logenergy_recon_subdivisions=9, longitude_recon_subdivisions=9, latitude_recon_subdivisions=9,
                 integration_method=logspace_simpson, parallelise_prior_construction=False,
                 mixture_tree:MTree=None, 
                 total_Nevents:int=None, pointing_dir=None, live_time=50*u.hr, 
                 irf_loglike=None):
        
        if pointing_dir is None:
            pointing_dir=np.array([0,0])*u.deg

        self.priors = priors
        self.default_true_binning_geometry = default_true_binning_geometry
        self.default_recon_binning_geometry = default_recon_binning_geometry


        self.logenergy_true_subdivisions = logenergy_true_subdivisions
        self.longitude_true_subdivisions = longitude_true_subdivisions
        self.latitude_true_subdivisions = latitude_true_subdivisions

        self.logenergy_recon_subdivisions = logenergy_recon_subdivisions
        self.longitude_recon_subdivisions = longitude_recon_subdivisions
        self.latitude_recon_subdivisions = latitude_recon_subdivisions


        self.integration_method = integration_method
        self.parallelise_prior_construction = parallelise_prior_construction
        self.mixture_tree = mixture_tree
        self.total_Nevents = total_Nevents
        self.pointing_dir = pointing_dir
        self.live_time = live_time
        self.irf_loglike = irf_loglike


    def construct_logprob_matrices(self, 
                                   logprob_funcs, logenergy_subdivisions=None, longitude_subdivisions=None, latitude_subdivisions=None, ):

        if logenergy_subdivisions is None:
            logenergy_subdivisions = self.logenergy_true_subdivisions

        if longitude_subdivisions is None:
            longitude_subdivisions = self.longitude_true_subdivisions

        if latitude_subdivisions is None:
            latitude_subdivisions = self.latitude_true_subdivisions


        energy_bin_edges    = self.default_true_binning_geometry.energy_edges
        longitude_bin_edges = self.default_true_binning_geometry.lon_edges
        latitude_bin_edges  = self.default_true_binning_geometry.lat_edges

        log_prob_arrays = [[]]*len(logprob_funcs)
        for logfunc_idx in tqdm(range(len(logprob_funcs)), desc='Constructing log probability arrays for sampling'):
            log_prob_array = -np.inf*np.zeros(shape=self.default_true_binning_geometry.axes_dim)
            for energy_idx in range(len(energy_bin_edges)-1):
                for lon_idx in range(len(longitude_bin_edges)-1):
                    for lat_idx in range(len(latitude_bin_edges)-1):
                        log_prob_array[energy_idx, lon_idx, lat_idx] = self._integrate_bin(logprob_funcs[logfunc_idx],
                                                                                                    energy_min=energy_bin_edges[energy_idx],
                                                                                                    energy_max=energy_bin_edges[energy_idx+1],
                                                                                                    longitude_min = longitude_bin_edges[lon_idx],
                                                                                                    longitude_max = longitude_bin_edges[lon_idx+1],
                                                                                                    latitude_min = latitude_bin_edges[lat_idx],
                                                                                                    latitude_max = latitude_bin_edges[lat_idx+1],
                                                                                                    log_energy_subdivision = logenergy_subdivisions,
                                                                                                    longitude_subdivisions = longitude_subdivisions,
                                                                                                    latitude_subdivisions = latitude_subdivisions,)
            
            
            log_prob_arrays[logfunc_idx] = log_prob_array
                        

        return log_prob_arrays



        



        
    def _integrate_bin(self, logfunc, 
                       energy_max, energy_min, 
                       longitude_max, longitude_min, 
                       latitude_max, latitude_min, 
                       log_energy_subdivision, longitude_subdivisions, latitude_subdivisions, 
                       **kwargs):

        _temp_energy_axis = np.logspace(np.log10(energy_min.value), np.log10(energy_max.value), log_energy_subdivision)*energy_max.unit
        _temp_longitude_axis = np.linspace(longitude_min.value, longitude_max.value, longitude_subdivisions)*longitude_max.unit
        _temp_latitude_axis = np.linspace(latitude_min.value, latitude_max.value, latitude_subdivisions)*latitude_max.unit

        axis_meshgrid = np.meshgrid(_temp_energy_axis, _temp_longitude_axis, _temp_latitude_axis, indexing='ij')

        flattened_axes = [axis.flatten() for axis in axis_meshgrid]

        binned_logprob_array = logfunc(*flattened_axes, **kwargs).reshape(axis_meshgrid[0].shape)

        integrated_bin_value = iterate_logspace_integration(binned_logprob_array, 
                                                            axes=[_temp_energy_axis.value, _temp_longitude_axis.value, _temp_latitude_axis.value,],
                                                            axisindices=[0, 1, 2],
                                                            logspace_integrator=self.integration_method)

        return integrated_bin_value





    def sample_true_log_pmf(self, Nsamples:int, integrated_logpmf):
        
        return integral_inverse_transform_sampler(integrated_logpmf, axes=self.default_true_binning_geometry.axes, Nsamples=Nsamples)
    


    def sample_true(self, total_Nsamples, mixture_tree=None):

        if total_Nsamples is None:
            total_Nsamples = self.total_Nevents

        if mixture_tree is None:
            mixture_tree = self.mixture_tree

        effective_Nsamples_by_prior = {prior_name:total_Nsamples*prior_frac for prior_name, prior_frac in mixture_tree.leaf_values.items()}


        prior_samples = GammaObs(binning_geometry=self.default_true_binning_geometry, 
                                 name="True IntraBin Samples",
                                 pointing_dirs=[self.pointing_dir],
                                 live_times=[self.live_time],)
        prior_logpmfs = self.construct_logprob_matrices(logprob_funcs=self.priors)

        self.prior_logpmfs = prior_logpmfs # Incase a user wants to query them after execution

        for prior, prior_logpmf in zip(self.priors, prior_logpmfs):
            raw_samples = self.sample_true_log_pmf(Nsamples=effective_Nsamples_by_prior[prior.name], integrated_logpmf=prior_logpmf)
            single_prior_obs=GammaObs(energy=raw_samples[0],
                                    lon=raw_samples[1],
                                    lat=raw_samples[2],
                                    binning_geometry=self.default_true_binning_geometry,
                                    pointing_dirs=[self.pointing_dir]*len(raw_samples[0]),
                                    live_times=[self.live_time]*len(raw_samples[0]))

            prior_samples+=single_prior_obs


        return prior_samples
    

    def sample_recon(self, true_obs, irf_loglike=None, logenergy_subdivisions=None, longitude_subdivisions=None, latitude_subdivisions=None, 
                     logenergy_radii=1, longitude_radii=1, latitude_radii=1):
        if irf_loglike is None:
            irf_loglike = self.irf_loglike


        if logenergy_subdivisions is None:
            logenergy_subdivisions = self.logenergy_recon_subdivisions

        if longitude_subdivisions is None:
            longitude_subdivisions = self.longitude_recon_subdivisions
        if latitude_subdivisions is None:
            latitude_subdivisions = self.latitude_recon_subdivisions

        recon_energies = []
        recon_lons = []
        recon_lats = []

        true_value_information_iterable = tqdm(zip(*true_obs.binned_unique_coordinate_data[0], true_obs.binned_unique_coordinate_data[1]), 
                                               desc="Sampling recon values", total=len(true_obs.binned_unique_coordinate_data[1]))

        for true_energy, true_longitude, true_latitude, event_num in true_value_information_iterable:
            # axes_meshes = np.meshgrid(true_energy, true_longitude, true_latitude, *self.default_recon_binning_geometry.axes, indexing='ij')
            # flattened_axes_meshes = [axis.flatten() for axis in axes_meshes]
            # irf_loglike_values = irf_loglike(*flattened_axes_meshes, pointing_dir=self.pointing_dir).reshape(self.default_recon_binning_geometry.axes_dim)

            irf_loglike_values = self.construct_constrained_logirf_matrix(
                irf_loglike=irf_loglike,
                true_energy=true_energy, true_lon=true_longitude, true_lat=true_latitude, 
                logenergy_subdivisions=logenergy_subdivisions, 
                longitude_subdivisions=longitude_subdivisions, 
                latitude_subdivisions=latitude_subdivisions,
                logenergy_radii=logenergy_radii, longitude_radii=longitude_radii, latitude_radii=latitude_radii

            )

            recon_energy, recon_lon, recon_lat = integral_inverse_transform_sampler(irf_loglike_values, axes=self.default_recon_binning_geometry.axes, Nsamples=event_num)


            recon_energies.extend(recon_energy.value)
            recon_lons.extend(recon_lon.value)
            recon_lats.extend(recon_lat.value)


        return GammaObs(energy=np.array(recon_energies)*self.default_recon_binning_geometry.energy_axis.unit, 
                        lon=recon_lons*self.default_recon_binning_geometry.lon_axis.unit, 
                        lat=recon_lats*self.default_recon_binning_geometry.lat_axis.unit, 
                        binning_geometry=self.default_recon_binning_geometry, irf_loglike=irf_loglike,
                        pointing_dirs=[self.pointing_dir], live_times=[self.live_time])
        



    def construct_constrained_logirf_matrix(self, 
                                              irf_loglike, 
                                              true_energy, true_lon, true_lat,
                                              logenergy_subdivisions, longitude_subdivisions, latitude_subdivisions, 
                                              logenergy_radii, longitude_radii, latitude_radii):

        energy_bin_edges    = self.default_recon_binning_geometry.energy_edges
        longitude_bin_edges = self.default_recon_binning_geometry.lon_edges
        latitude_bin_edges  = self.default_recon_binning_geometry.lat_edges


        log_prob_array = -np.inf*np.zeros(shape=self.default_recon_binning_geometry.axes_dim)

        energy_condition = np.where(np.abs(np.log10(energy_bin_edges.value)-np.log10(true_energy.value))<logenergy_radii, True, False)[:-1]
        lon_condition = np.where(np.abs(longitude_bin_edges.value-true_lon.value)<longitude_radii, True, False)[:-1]
        lat_condition = np.where(np.abs(latitude_bin_edges.value-true_lat.value)<latitude_radii, True, False)[:-1]

        energy_slice = energy_condition*np.arange(len(energy_condition))
        lon_slice = lon_condition*np.arange(len(lon_condition))
        lat_slice = lat_condition*np.arange(len(lat_condition))


        log_prob_array[*np.meshgrid(energy_slice, lon_slice, lat_slice, indexing='ij')] = self._integrate_irf_bin(
            irf_loglike,
            energy_mins     = energy_bin_edges[energy_slice],
            energy_maxs     = energy_bin_edges[energy_slice+1],
            longitude_mins  = longitude_bin_edges[lon_slice],
            longitude_maxs  = longitude_bin_edges[lon_slice+1],
            latitude_mins   = latitude_bin_edges[lat_slice],
            latitude_maxs   = latitude_bin_edges[lat_slice+1],
            log_energy_subdivisions = logenergy_subdivisions,
            longitude_subdivisions  = longitude_subdivisions,
            latitude_subdivisions   = latitude_subdivisions,
            true_energy = true_energy,
            true_lon    = true_lon,
            true_lat    = true_lat,
            )
        
        return log_prob_array



        
    def _integrate_irf_bin(self, logfunc, 
                energy_maxs, energy_mins, 
                longitude_maxs, longitude_mins, 
                latitude_maxs, latitude_mins, 
                log_energy_subdivisions, longitude_subdivisions, latitude_subdivisions, 
                true_energy=None, true_lat=None, true_lon=None,
                **kwargs):

        # times = []
        # times.append(time.perf_counter())

        _temp_energy_axes = np.asanyarray([np.logspace(np.log10(energy_min.value), np.log10(energy_max.value), log_energy_subdivisions) for energy_min, energy_max in zip(energy_mins, energy_maxs)]).T*energy_maxs[0].unit

        _temp_longitude_axes = np.asanyarray([np.linspace(longitude_min.value, longitude_max.value, longitude_subdivisions) for longitude_min, longitude_max in zip(longitude_mins, longitude_maxs)]).T*longitude_maxs[0].unit

        _temp_latitude_axes = np.asanyarray([np.linspace(latitude_min.value, latitude_max.value, latitude_subdivisions) for latitude_min, latitude_max in zip(latitude_mins, latitude_maxs)]).T*latitude_maxs[0].unit

        axis_meshgrid = np.meshgrid(_temp_energy_axes, _temp_longitude_axes, _temp_latitude_axes, true_energy, true_lat, true_lon, indexing='ij')
        
        # times.append(time.perf_counter())

        flattened_axes = [axis.flatten() for axis in axis_meshgrid]

        # times.append(time.perf_counter())

        binned_logprob_array = logfunc(*flattened_axes, **kwargs).reshape((*_temp_energy_axes.shape, *_temp_longitude_axes.shape, *_temp_latitude_axes.shape))

        # times.append(time.perf_counter())


        integrated_bin_value = []
        for energy_idx, energy_val in enumerate(energy_mins):
            integrated_bin_value_for_energy = []
            for lon_idx, lon_val in enumerate(longitude_mins):
                integrated_bin_value_for_lon = []
                for lat_idx, lat_val in enumerate(latitude_mins):
                    integrated_bin_value_for_lon.append(iterate_logspace_integration(binned_logprob_array[:, energy_idx, :, lon_idx, :, lat_idx], 
                                                                    axes=[_temp_energy_axes[:, energy_idx].value, _temp_longitude_axes[:, lon_idx].value, _temp_latitude_axes[:, lat_idx].value,],
                                                                    axisindices=[0, 1, 2], 
                                                                    logspace_integrator=self.integration_method))
                integrated_bin_value_for_energy.append(integrated_bin_value_for_lon)
            integrated_bin_value.append(integrated_bin_value_for_energy)

        # times.append(time.perf_counter())
        # print(np.diff(np.array(times)))

        return integrated_bin_value
