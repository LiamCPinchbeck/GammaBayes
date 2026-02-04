import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from torch.distributions import Categorical
import numpy as np

class DiscreteLogPrior:
    arg_constraints = {}
    support = constraints.real
    has_enumerate_support = False

    def __init__(self, binning_geometry, rate_tensor_func=None, name=None, rate_tensor_logfunc=None):
        """
        energy_edges: assumed to be in linear space, log10 spacing is applied internally. Presumes that 
        the inputs are either linearly spaced or have the appropriate jacobian applied.
        """
        self.binning_geometry = binning_geometry
        if rate_tensor_logfunc is not None:
            self.log_rate = rate_tensor_logfunc
        else:
            self.log_rate = self._log_rate

        if rate_tensor_func is not None:
            self.rate_tensor_func = rate_tensor_func
        else:
            self.rate_tensor_func = self._exp_log_rate

        if self.rate_tensor_func is None and self.log_rate is None:
            raise ValueError("Either 'rate_tensor_func' or 'rate_tensor_logfunc' must be given.")


        self.name = name
        self._cached_probs = None
        self._cached_params = None


    def __repr__(self) -> str:
        """
        String representation of the DiscreteLogPrior instance.

        Returns:
            str: A description of the instance including its name, logfunction type, input units, and axes names.
        """
        description = f"Discrete log prior class\n{'-' * 20}\n" \
                      f"Name: {self.name}\n" \
                      f"Logfunction type: {type(self.log_rate).__name__}\n"
        return description


    def __call__(self, *args, **kwargs):

        output = self.log_rate(*args, **kwargs)
        return output


    def _evaluate(self, **params):
        key = tuple(sorted((k, self._tensor_hash(v)) for k, v in params.items()))

        if key == self._cached_params and self._cached_rates is not None:
            rates = self._cached_rates
            norm  = self._cached_norm
        else:
            rates = self.rate_tensor_func(self.binning_geometry.grid, **params).float()
            norm = rates.sum()
            self._cached_rates  = rates
            self._cached_norm   = norm
            self._cached_params = key

        return rates

    def _get_categorical(self, **params):

        rates = self._evaluate(**params)
        norm = self._cached_norm

        return Categorical(probs=rates/norm)

    def _tensor_hash(self, x):
        """Create a hashable summary of a tensor to detect changes."""
        if torch.is_tensor(x):
            return (x.device.type, tuple(x.shape), float(x.sum()))
        else:
            return x


    def sample(self, norm, normalise=False, **params):

        rates = torch.exp(self.eval_log_on_geom(**params))
        
        if normalise:
            rates/=rates.sum()

        poiss_dist = dist.Poisson(rate=norm*rates)

        samples = poiss_dist.sample()

        return samples

    def sample_categorical(self, num_samples, **params):
        categorical_dist = self._get_categorical(**params)
        categorical_dist_samples_indices = torch.unravel_index(categorical_dist.sample((num_samples,)), shape=self.binning_geometry.shape)
        categorical_dist_samples = torch.stack([axis[samples] for axis, samples in zip(self.binning_geometry.axes, categorical_dist_samples_indices)], dim=1)


        categorical_dist_samples_hist = torch.histogramdd(categorical_dist_samples, self.binning_geometry.axes_edges).hist

        return categorical_dist_samples_hist


    def log_prob(self, value, **params):
        indices = self.binning_geometry.values_to_grid(value)

        cat = self._get_categorical(**params)

        return cat.log_prob(indices)


    def _log_rate(self, value, **params):
        rates = self.rate_tensor_func(value, **params).float()
        return torch.log(rates)

    def eval_log_on_geom(self, **params):
        return self.log_rate(
            self.binning_geometry.shaped_grid, 
            **params
            )

    def _exp_log_rate(self, *args, **kwargs):
        return torch.exp(self.log_rate(*args, **kwargs))


    def log_normalisation(self, log_prior_values, parameters={}, *args, **kwargs):
        if (log_prior_values is []) | (log_prior_values is None):
            log_prior_values = self.eval_log_on_geom(**parameters)

        # Annoying PyTorch Error where if you specify None as in the documentation it raises an error
            # This ensure that all the axes are reduced, regardless of dimension
        dims = tuple(torch.arange(log_prior_values.ndim).numpy())

        return torch.logsumexp(log_prior_values, dim=dims)


    def peek(self, fig=None, axes=None, pcm_kwargs={}, plot_kwargs={}, fig_kwargs = {}, **params):
        if 'norm' not in pcm_kwargs:
            pcm_kwargs['norm'] = 'log'

        if 'yscale' not in plot_kwargs:
            plot_kwargs['yscale'] = 'log'

        if 'xscale' not in plot_kwargs:
            plot_kwargs['xscale'] = 'log'
        if 'figsize' not in fig_kwargs:
            fig_kwargs['figsize'] = (18, 10)


        from matplotlib import pyplot as plt

        full_mat = self.eval_log_on_geom(**params)

        energy_mat = torch.logsumexp(full_mat, dim=(1,2)).exp().detach().cpu().numpy()
        lonlat_mat = torch.logsumexp(full_mat, dim=0).exp().detach().cpu().numpy()

        plotting_energy_axis = self.binning_geometry.energy_axis.clone().detach().cpu().numpy()
        plotting_lon_axis = self.binning_geometry.lon_axis.clone().detach().cpu().numpy()
        plotting_lat_axis = self.binning_geometry.lat_axis.clone().detach().cpu().numpy()

        if fig is None or axes is None:
            fig, axes = plt.subplots(2, 2, **fig_kwargs)

            axes = axes.flatten()


        full_mat = full_mat.detach().cpu().numpy()

        axes[0].plot(plotting_energy_axis, energy_mat)
        axes[0].set(
            xlabel="True Energy [TeV]",
            **plot_kwargs
        )

        pcm = axes[1].pcolormesh(plotting_lon_axis, plotting_lat_axis, lonlat_mat.T, **pcm_kwargs)

        plt.colorbar(pcm, ax=axes[1])
        axes[1].set(
            xlabel="Galactic Longitude [deg]",
            ylabel="Galactic Latitude [deg]",
            aspect='equal',
        )


        axes[2].plot(plotting_energy_axis, np.exp(full_mat[:, full_mat.shape[1]//2, full_mat.shape[2]//2,]))
        axes[2].set(
            xlabel="True Energy [TeV]",
            **plot_kwargs
        )

        pcm = axes[3].pcolormesh(plotting_lon_axis, plotting_lat_axis, np.exp(full_mat[np.abs(plotting_energy_axis - 1.).argmin(), :, :].T), **pcm_kwargs)

        plt.colorbar(pcm, ax=axes[3])
        axes[3].set(
            xlabel="Galactic Longitude [deg]",
            ylabel="Galactic Latitude [deg]",
            aspect='equal',
        )

        plt.tight_layout()

        return fig, axes

