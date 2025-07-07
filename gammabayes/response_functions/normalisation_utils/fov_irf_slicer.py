import numpy as np
from gammabayes import GammaBinning, ParameterSet
from gammabayes.utils.integration import iterate_logspace_integration
from tqdm import tqdm
from astropy import units as u

from icecream import ic


class FOV_IRF_Norm:
    def __init__(self, 
                 true_binning_geometry: GammaBinning, 
                 recon_binning_geometry: GammaBinning, 
                 original_norm_matrix_pointing_dir: np.ndarray[u.Quantity],
                 new_pointing: np.ndarray[u.Quantity],
                 pointing_dirs: list[np.ndarray[u.Quantity]]|tuple[np.ndarray[u.Quantity]]=None, observation_time:u.Quantity=None,
                 irf_loglike=None, 
                 log_edisp_norm_matrix:np.ndarray=None,
                 log_psf_norm_matrix:np.ndarray=None,
                 irf_norm_matrix:np.ndarray=None,
                 edisp_parameters:ParameterSet=None,
                 psf_parameters:ParameterSet=None,
                 ):
        """Assumes linear spatial binning. Energy binning doesn't matter for this class as it isn't impacted."""
        self.observation_time=observation_time
        self.true_binning_geometry = true_binning_geometry
        self.recon_binning_geometry = recon_binning_geometry
        self.original_norm_matrix_pointing_dir = original_norm_matrix_pointing_dir
        self.pointing_dirs=pointing_dirs
        self.irf_loglike = irf_loglike
        self.pointing_dir = new_pointing

        self.edisp_parameters          = ParameterSet(edisp_parameters)
        self.psf_parameters          = ParameterSet(psf_parameters)


        self._setup_irf_parameter_sets()

        if not(any(np.array_equal(self.pointing_dir, arr) for arr in list(self.pointing_dirs))):
            self.pointing_dirs.append(self.pointing_dir)



        self.original_log_psf_norm_matrix = log_psf_norm_matrix
        self.original_log_edisp_norm_matrix = log_edisp_norm_matrix
        self.original_irf_norm_matrix = irf_norm_matrix


        if not(self.original_log_edisp_norm_matrix is None) and not(self.original_log_psf_norm_matrix is None):
            self.use_irf_norm_matrix = False
            self.empty_input = False
        elif not(self.original_irf_norm_matrix is None):
            self.use_irf_norm_matrix = True
            self.empty_input = False
        else:
            self.use_irf_norm_matrix = False
            self.empty_input = True


        ###  
        # x ==>previous corners
        # o ==>new corners
        # + ==>previous pointing direction
        # <> ==>new pointing direction
        # || and =  ==> effective
        #  x-----------x
        #  |           |
        #  |     +     |
        #  |           |
        #  x-----------x
        #        |
        #        |
        #        |
        #       \  /
        #        \/

        #  o================---o
        # ||   |      ||   |   |
        # ||---x<>----||---x---|
        # ||   |      ||   |   |
        # ||===|=====+||   |   |
        #  |   |           |   |
        #  |---x-----------x---|
        #  |   |           |   |
        #  o-------------------o

        self.refresh()


    def _setup_irf_parameter_sets(self):

        self.edisp_parameter_axes_by_name = {parameter.name:parameter.axis for parameter in self.edisp_parameters._dict_of_parameters_by_name.values()}
        self.psf_parameter_axes_by_name = {parameter.name:parameter.axis for parameter in self.psf_parameters._dict_of_parameters_by_name.values()}

        self.num_edisp_parameters = len(self.edisp_parameter_axes_by_name)
        self.num_psf_parameters = len(self.psf_parameter_axes_by_name)

        self.edisp_axes_dims = [len(axis) for axis in self.edisp_parameter_axes_by_name.values()]
        self.psf_axes_dims = [len(axis) for axis in self.psf_parameter_axes_by_name.values()]




    def refresh(self, pointing_dir=None):
        "Used to reset the default pointing direction (and thus observation window) of normalisation matrices"

        if pointing_dir is not None:
            self.pointing_dir = pointing_dir

        self._setup_irf_parameter_sets()

        self.left_longitude_buffer, self.right_longitude_buffer, self.lower_latitude_buffer, self.upper_latitude_buffer = self._define_buffers()

        new_true_angular_axes, (left_true_buffer_indices, right_true_buffer_indices, lower_true_buffer_indices, upper_true_buffer_indices), _ = self._define_new_axes(self.true_binning_geometry, self.left_longitude_buffer, self.right_longitude_buffer, self.lower_latitude_buffer, self.upper_latitude_buffer)



        self.buffered_true_binning_geometry  = GammaBinning(self.true_binning_geometry.energy_axis, *new_true_angular_axes)


        if right_true_buffer_indices>0:
            right_buffer_slice_val = -right_true_buffer_indices
            
        else:
            right_buffer_slice_val = None

        if upper_true_buffer_indices>0:
            upper_buffer_slice_val = -upper_true_buffer_indices
        else:

            upper_buffer_slice_val = None


        self.default_buffer_window_values = ((None, None), (left_true_buffer_indices, right_true_buffer_indices), (lower_true_buffer_indices, upper_true_buffer_indices))



        if not(self.use_irf_norm_matrix):
            self.buffered_log_edisp_norm = np.full(fill_value=np.nan, shape=(*self.buffered_true_binning_geometry.axes_dim, *self.edisp_axes_dims))
            self.buffered_log_psf_norm = np.full(fill_value=np.nan, shape=(*self.buffered_true_binning_geometry.axes_dim, *self.psf_axes_dims))
            if not self.empty_input:
                self.buffered_log_psf_norm[:, left_true_buffer_indices:right_buffer_slice_val, lower_true_buffer_indices:upper_buffer_slice_val, ...] = self.original_log_psf_norm_matrix
                self.buffered_log_edisp_norm[:, left_true_buffer_indices:right_buffer_slice_val, lower_true_buffer_indices:upper_buffer_slice_val, ...] = self.original_log_edisp_norm_matrix
            
        else:
            self.buffered_log_irf_norm = np.full(fill_value=np.nan, shape=self.buffered_true_binning_geometry.axes_dim)
            self.buffered_log_irf_norm[:, left_true_buffer_indices:right_buffer_slice_val, lower_true_buffer_indices:upper_buffer_slice_val] = self.original_irf_norm_matrix


        self._fill_norm_matrices()

        self._refresh_buffer_window()


    def _fill_norm_matrices(self):


        if not(self.use_irf_norm_matrix):
            isnan_boolean_matrix = np.isnan(self.buffered_log_edisp_norm)
        else:
            isnan_boolean_matrix = np.isnan(self.buffered_log_irf_norm)


        self.irf_loglike.pointing = self.original_norm_matrix_pointing_dir


        for energy_slice_idx, nan_index_energy_slice in tqdm(enumerate(isnan_boolean_matrix[:, :, :, *[0]*self.num_edisp_parameters]), desc="Filling in buffer", total=len(isnan_boolean_matrix)):

            true_energy_val = self.buffered_true_binning_geometry.energy_axis[energy_slice_idx]

            if not(self.use_irf_norm_matrix):        
                unfiltered_spatial_meshes = np.meshgrid(*self.buffered_true_binning_geometry.spatial_axes, indexing='ij')

                sliced_spatial_meshes = [spatial_mesh[nan_index_energy_slice].reshape(nan_index_energy_slice.shape) for spatial_mesh in unfiltered_spatial_meshes]

                normed_edisp_values = self._edisp_normalisation(self.recon_binning_geometry, true_energy_val, *sliced_spatial_meshes, edisp_parameter_axes=self.edisp_parameter_axes_by_name)

                reshaped_normed_edisp_values = normed_edisp_values.squeeze().reshape((-1, *self.edisp_axes_dims))
                
                self.buffered_log_edisp_norm[energy_slice_idx, nan_index_energy_slice, ...] = reshaped_normed_edisp_values

            for lon_slice_idx, nan_index_energy_lon_slice in enumerate(nan_index_energy_slice):

                true_lon_val = self.buffered_true_binning_geometry.lon_axis[lon_slice_idx]
                

                if not(self.use_irf_norm_matrix) and (np.sum(nan_index_energy_lon_slice)>0):

                    true_lat_vals = self.buffered_true_binning_geometry.lat_axis[nan_index_energy_lon_slice]
                    
                    normed_psf_values = self._psf_normalisation(self.recon_binning_geometry,  true_energy_val, true_lon_val, true_lat_vals, psf_parameter_axes=self.psf_parameter_axes_by_name)

                    self.buffered_log_psf_norm[energy_slice_idx, lon_slice_idx, nan_index_energy_lon_slice, ...]   = normed_psf_values


                else:
                    # self.buffered_log_irf_norm[energy_idx, lon_idx, lat_idx]   = self.irf_loglike(energy=energy_vals, lon=lon_vals, lat=lat_vals)
                    raise Exception("Oh no! You've hit a condition that I couldn't be bothered coding up yet!")
                


        # Some values contained within normalisation give 0. because that's where the MCMC cuts off. This is very annoying when wanting to normalise these values
            # But because they are all zero (or at least normalise to it) then we can just leave them as they are (i.e. we use 0/1 instead of 0/0).
        if not(self.use_irf_norm_matrix):
            self.buffered_log_psf_norm[np.isneginf(self.buffered_log_psf_norm)] = 0
            self.buffered_log_edisp_norm[np.isneginf(self.buffered_log_edisp_norm)] = 0






    def _edisp_normalisation(self, recon_binning_geometry, true_energy_val, true_lon_mesh_vals, true_lat_mesh_vals, edisp_parameter_axes=None):
        if edisp_parameter_axes is None:
            edisp_parameter_axes = {}


        
        recon_energy_vals = recon_binning_geometry.energy_axis

        recon_energy_mesh, true_energy_axis_mesh, true_lon_axis_mesh, *edisp_parameter_axes_meshes = np.meshgrid(recon_energy_vals, 
                                                                                   true_energy_val, true_lon_mesh_vals.flatten(), 
                                                                                   *edisp_parameter_axes.values(), indexing='ij')


        edisp_parameter_axes_meshes_flattened = {kwd:parameter_axis_mesh.flatten() for kwd, parameter_axis_mesh in zip(edisp_parameter_axes.keys(), edisp_parameter_axes_meshes)}

        true_lat_axis_mesh = np.broadcast_to(true_lat_mesh_vals.flatten()[np.newaxis, np.newaxis, :, *[np.newaxis]*len(edisp_parameter_axes_meshes)], recon_energy_mesh.shape)


        if not(hasattr(true_lat_axis_mesh, 'unit')):
            if hasattr(true_lat_mesh_vals, 'unit'):
                true_lat_axis_mesh = true_lat_axis_mesh*true_lat_mesh_vals.unit
            else:
                true_lat_axis_mesh = true_lat_axis_mesh*true_lon_mesh_vals.unit


        log_edisp_values = self.irf_loglike.log_edisp(
                recon_energy=recon_energy_mesh.flatten(), 
                true_energy=true_energy_axis_mesh.flatten(),  
                true_lon=true_lon_axis_mesh.flatten(),  
                true_lat=true_lat_axis_mesh.flatten(),
                **edisp_parameter_axes_meshes_flattened
                ).reshape(recon_energy_mesh.shape)
        
        normed_edisp_values = iterate_logspace_integration(
            logy=np.squeeze(log_edisp_values), 
            axes=(recon_energy_vals.value,), 
            axisindices=[0,]).reshape((*true_lon_mesh_vals.shape, *self.edisp_axes_dims))
        
        return normed_edisp_values
    


    def _psf_normalisation(self, recon_binning_geometry, true_energy_val, true_lon_val, true_lat_vals, psf_parameter_axes=None):
        if psf_parameter_axes is None:
            psf_parameter_axes = {}



        recon_lon_vals = recon_binning_geometry.lon_axis
        recon_lat_vals = recon_binning_geometry.lat_axis

        recon_lon_mesh,recon_lat_mesh, true_energy_mesh, true_lon_mesh, true_lat_mesh, *psf_parameter_axes_meshes = np.meshgrid(
            recon_lon_vals, recon_lat_vals, true_energy_val, true_lon_val, true_lat_vals, *psf_parameter_axes.values(), indexing='ij')

        psf_parameter_axes_meshes_flattened_dict = {kwd:parameter_axis_mesh.flatten() for kwd, parameter_axis_mesh in zip(psf_parameter_axes.keys(), psf_parameter_axes_meshes)}


        log_psf_values = self.irf_loglike.log_psf(
                recon_lon=recon_lon_mesh.flatten(), 
                recon_lat=recon_lat_mesh.flatten(), 
                true_energy=true_energy_mesh.flatten(),  
                true_lon=true_lon_mesh.flatten(),  
                true_lat=true_lat_mesh.flatten(),
                **psf_parameter_axes_meshes_flattened_dict,
                ).reshape(recon_lon_mesh.shape)
        
        normed_psf_values = iterate_logspace_integration(
            logy=np.squeeze(log_psf_values), 
            axes=(recon_lon_vals.value, recon_lat_vals.value,), 
            axisindices=[0,1]).reshape((*true_lat_vals.shape, *self.psf_axes_dims))
        
        
        return normed_psf_values






    def _define_new_axes(self, binning_geometry:GammaBinning, left_longitude_buffer, right_longitude_buffer, lower_latitude_buffer, upper_latitude_buffer):

        lon_res = self.true_binning_geometry.lon_res
        lat_res = self.true_binning_geometry.lat_res



        new_min_longitude = binning_geometry.lon_axis[0]-int(np.round(left_longitude_buffer/lon_res))*lon_res
        new_max_longitude = binning_geometry.lon_axis[-1]+int(np.round(right_longitude_buffer/lon_res))*lon_res
        new_min_latitude = binning_geometry.lat_axis[0]-int(np.round(lower_latitude_buffer/lat_res))*lat_res
        new_max_latitude = binning_geometry.lat_axis[-1]+int(np.round(upper_latitude_buffer/lat_res))*lat_res


        left_buffer_values  = np.arange(new_min_longitude.value, binning_geometry.lon_axis[0].value, lon_res.value)*lon_res.unit

        if not np.isclose(left_buffer_values.value, binning_geometry.lon_axis[0].value).any():
            left_buffer_values = np.append(left_buffer_values.value, binning_geometry.lon_axis[0].value)*lon_res.unit

        right_buffer_values = np.arange(binning_geometry.lon_axis[-1].value, new_max_longitude.value, lon_res.value)*lon_res.unit

        if not np.isclose(right_buffer_values.value, new_max_longitude.value).any():
            right_buffer_values = np.append(right_buffer_values.value, new_max_longitude.value)*lon_res.unit
        

        left_buffer_indices = len(left_buffer_values)-1
        right_buffer_indices = len(right_buffer_values)-1



        lower_buffer_values = np.arange(new_min_latitude.value, binning_geometry.lat_axis[0].value, lat_res.value)*lat_res.unit
        upper_buffer_values = np.arange(binning_geometry.lat_axis[-1].value, new_max_latitude.value, lat_res.value)*lat_res.unit


        if not np.isclose(lower_buffer_values.value, binning_geometry.lat_axis[0].value).any():
            lower_buffer_values = np.append(lower_buffer_values.value, binning_geometry.lat_axis[0].value)*lat_res.unit


        if not np.isclose(upper_buffer_values.value, new_max_latitude.value).any():
            upper_buffer_values = np.append(upper_buffer_values.value, new_max_latitude.value)*lat_res.unit


        lower_buffer_indices = len(lower_buffer_values)-1
        upper_buffer_indices = len(upper_buffer_values)-1


        


        new_longitude_axis = np.arange(new_min_longitude.value, new_max_longitude.value, lon_res.value)*lon_res.unit
        new_latitude_axis = np.arange(new_min_latitude.value, new_max_latitude.value, lat_res.value)*lat_res.unit

        if not np.isclose(new_longitude_axis.value, new_max_longitude.value).any():
            new_longitude_axis = np.sort(np.append(new_longitude_axis.value, new_max_longitude.value)*lon_res.unit)

        if not np.isclose(new_longitude_axis.value, new_min_longitude.value).any():
            new_longitude_axis = np.sort(np.append(new_longitude_axis.value, new_min_longitude.value)*lon_res.unit)

        if not np.isclose(new_latitude_axis.value, new_min_latitude.value).any():
            new_latitude_axis = np.sort(np.append(new_latitude_axis.value, new_min_latitude.value)*lat_res.unit)

        if not np.isclose(new_latitude_axis.value, new_max_latitude.value).any():

            new_latitude_axis = np.sort(np.append(new_latitude_axis.value, new_max_latitude.value)*lat_res.unit)

        return (new_longitude_axis, new_latitude_axis), (left_buffer_indices, right_buffer_indices, lower_buffer_indices, upper_buffer_indices), (left_buffer_values, right_buffer_values, lower_buffer_values, upper_buffer_values)


    def _refresh_buffer_window(self):

        lon_offset = int(round((self.original_norm_matrix_pointing_dir[0].value-self.pointing_dir[0].value)/self.buffered_true_binning_geometry.lon_res.value))
        lat_offset = int(round((self.original_norm_matrix_pointing_dir[1].value-self.pointing_dir[1].value)/self.buffered_true_binning_geometry.lat_res.value))



        default_lon_vals = self.default_buffer_window_values[1]
        default_lat_vals = self.default_buffer_window_values[2]

        lon_start = (default_lon_vals[0] + lon_offset)  

        if default_lon_vals[1] is None:
            lon_stop = -lon_offset

        else:
            lon_stop = -(default_lon_vals[1] - lon_offset)

        lat_start = (default_lat_vals[0] + lat_offset)  

        if default_lat_vals[1] is None:
            lat_stop = -lat_offset

        else:
            lat_stop = -(default_lat_vals[1] - lat_offset)

        if lon_stop == 0:
            lon_stop = None

        if lat_stop == 0:
            lat_stop = None

        self.buffer_window = (
            slice(None, None),
            slice(lon_start, lon_stop),
            slice(lat_start, lat_stop)
        )





    def _define_buffers(self):


        pointing_longitudes = [pointing[0] for pointing in self.pointing_dirs]
        pointing_latitudes  = [pointing[1] for pointing in self.pointing_dirs]

        min_pt_longitude = min(pointing_longitudes)
        min_pt_latitude  = min(pointing_latitudes)

        max_pt_longitude = max(pointing_longitudes)
        max_pt_latitude  = max(pointing_latitudes)

        # The pointing to the most towards the left of the observation window
        #    will decide how much buffer must be added to the right
        right_longitude_buffer = self.original_norm_matrix_pointing_dir[0]-min_pt_longitude
        if right_longitude_buffer<0:
            right_longitude_buffer = 0

        # The pointing to the most towards the bottom of the observation window
        #    will decide how much buffer must be added to the top
        upper_latitude_buffer = self.original_norm_matrix_pointing_dir[1]-min_pt_latitude
        if upper_latitude_buffer<0:
            upper_latitude_buffer = 0


        # The pointing to the most towards the right of the observation window
        #    will decide how much buffer must be added to the left

        left_longitude_buffer = max_pt_longitude-self.original_norm_matrix_pointing_dir[0]
        if left_longitude_buffer<0:
            left_longitude_buffer = 0
        # The pointing to the most towards the top of the observation window
        #    will decide how much buffer must be added to the bottom
        lower_latitude_buffer = max_pt_latitude-self.original_norm_matrix_pointing_dir[1]
        if lower_latitude_buffer<0:
            lower_latitude_buffer = 0

        return left_longitude_buffer, right_longitude_buffer, lower_latitude_buffer, upper_latitude_buffer
    

    def peek(self, subplots_kwargs, pcolormesh_kwargs, **kwargs):

        from matplotlib import pyplot as plt
        from matplotlib import patches
        from matplotlib.colors import LogNorm

        subplots_kwargs.update(kwargs)


        if 'norm' not in pcolormesh_kwargs:
            pcolormesh_kwargs['norm'] = 'log'





        # if not self.use_irf_norm_matrix:
        #     integrated_irf_norm = iterate_logspace_integration(logy = self.buffered_log_edisp_norm+self.buffered_log_psf_norm, 
        #                                                     axes=(self.buffered_true_binning_geometry.energy_axis.value,), 
        #                                                     axisindices=[0])
        # else:
        #     integrated_irf_norm = iterate_logspace_integration(logy = self.buffered_log_irf_norm, 
        #                                                     axes=(self.buffered_true_binning_geometry.energy_axis.value,), 
        #                                                     axisindices=[0])




        fig, axes = plt.subplots(1, 3, **subplots_kwargs)

        len_energy_axis = len(self.buffered_true_binning_geometry.energy_axis)

        for axis_idx, axis in enumerate(axes):

            pcm = axis.pcolormesh(self.buffered_true_binning_geometry.lon_axis.value, 
                            self.buffered_true_binning_geometry.lat_axis.value, 
                            np.exp(self.buffered_log_edisp_norm[int(axis_idx/3*len_energy_axis), :, :]+self.buffered_log_psf_norm[int(axis_idx/3*len_energy_axis), :, :]), **pcolormesh_kwargs)
            plt.colorbar(mappable=pcm, label="Normalisation Values", ax=axis)
            axis.set_xlabel(r"Longitude ["+self.buffered_true_binning_geometry.lon_axis.unit.to_string('latex')+']')
            axis.set_ylabel(r"Latitude ["+self.buffered_true_binning_geometry.lat_axis.unit.to_string('latex')+']')

            longitude_start, longitude_end, latitude_start, latitude_end = *self.buffered_true_binning_geometry.lon_axis[self.buffer_window[1]][[0,-1]], *self.buffered_true_binning_geometry.lat_axis[self.buffer_window[2]][[0,-1]]

            width  = longitude_end - longitude_start
            height = latitude_end  - latitude_start

            buffer_window = patches.Rectangle((longitude_start.value, latitude_start.value), width.value, height.value, linewidth=1, edgecolor='tab:orange', facecolor='none')
            axis.add_patch(buffer_window)

            for pointing_dir in self.pointing_dirs:
                axis.scatter(*pointing_dir, c='tab:red')
            
            axis.scatter(*self.pointing_dir, c='tab:orange', marker='x', s=20)


            default_longitude_start, default_longitude_end, default_latitude_start, default_latitude_end = *self.buffered_true_binning_geometry.lon_axis[[*self.default_buffer_window_values[1]]], *self.buffered_true_binning_geometry.lat_axis[[*self.default_buffer_window_values[2]]]


            default_width  = default_longitude_end - default_longitude_start
            default_height = default_latitude_end  - default_latitude_start

            default_buffer_window = patches.Rectangle((default_longitude_start.value, default_latitude_start.value), 
                                                      default_width.value, default_height.value, 
                                                      linewidth=2, 
                                                      edgecolor='tab:blue', facecolor='none', linestyle='--')
            axis.add_patch(default_buffer_window)

            axis.scatter(*self.original_norm_matrix_pointing_dir, c='tab:blue', marker='x', s=20)


        plt.tight_layout()
        plt.show()
    


    def __getitem__(self, indices):

        if not self.use_irf_norm_matrix:
            return (self.buffered_log_edisp_norm[*self.buffer_window, ..., *[np.newaxis]*self.num_psf_parameters] \
                + self.buffered_log_psf_norm[*self.buffer_window, *[np.newaxis]*self.num_edisp_parameters, ...])[indices]
        else:
            return self.buffered_log_irf_norm[self.buffer_window][indices]
        

    def __add__(self, other):
        if not self.use_irf_norm_matrix:
            return (self.buffered_log_edisp_norm[*self.buffer_window, ..., *[np.newaxis]*self.num_psf_parameters] +
                    self.buffered_log_psf_norm[*self.buffer_window, *[np.newaxis]*self.num_edisp_parameters, ...]) + other
        else:
            return self.buffered_log_irf_norm[self.buffer_window] + other

    def __sub__(self, other):
        if not self.use_irf_norm_matrix:
            return (self.buffered_log_edisp_norm[*self.buffer_window, ..., *[np.newaxis]*self.num_psf_parameters] +
                    self.buffered_log_psf_norm[*self.buffer_window, *[np.newaxis]*self.num_edisp_parameters, ...]) - other
        else:
            return self.buffered_log_irf_norm[self.buffer_window] - other

    def __radd__(self, other):
        # Addition is commutative, so we can reuse __add__
        return self.__add__(other)

    def __rsub__(self, other):
        if not self.use_irf_norm_matrix:
            return other - (self.buffered_log_edisp_norm[*self.buffer_window, ..., *[np.newaxis]*self.num_psf_parameters] +
                            self.buffered_log_psf_norm[*self.buffer_window, *[np.newaxis]*self.num_edisp_parameters, ...])
        else:
            return other - self.buffered_log_irf_norm[self.buffer_window]
        


    def __array__(self, dtype=None):
        if not self.use_irf_norm_matrix:
            result = self.buffered_log_edisp_norm[*self.buffer_window, ..., *[np.newaxis]*self.num_psf_parameters] + \
                     self.buffered_log_psf_norm[*self.buffer_window, *[np.newaxis]*self.num_edisp_parameters, ...]
        else:
            result = self.buffered_log_irf_norm[self.buffer_window]
        
        if dtype is not None:
            return np.asarray(result, dtype=dtype)
        return np.asarray(result)

