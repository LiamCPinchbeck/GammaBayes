import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from torch.distributions import Categorical


class DiscreteLogPrior:
    arg_constraints = {}
    support = constraints.real
    has_enumerate_support = False

    def __init__(self, coord_geom, rate_tensor_func=None, rate_tensor_logfunc=None):
        """
        energy_edges: assumed to be in linear space, log10 spacing is applied internally. Presumes that 
        the inputs are either linearly spaced or have the appropriate jacobian applied.
        """
        self.coord_geom = coord_geom
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


        self._cached_probs = None
        self._cached_params = None


    def _evaluate(self, **params):
        key = tuple(sorted((k, self._tensor_hash(v)) for k, v in params.items()))

        if key == self._cached_params and self._cached_rates is not None:
            rates = self._cached_rates
            norm  = self._cached_norm
        else:
            rates = self.rate_tensor_func(self.coord_geom.grid, **params).float()
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

        rates = norm*torch.exp(self.eval_log_on_geom(**params))
        
        if normalise:
            rates/=rates.sum()

        poiss_dist = dist.Poisson(rate=rates)

        samples = poiss_dist.sample()

        return samples


    def log_prob(self, value, **params):
        indices = self.coord_geom.values_to_grid(value)

        cat = self._get_categorical(**params)

        return cat.log_prob(indices)


    def _log_rate(self, value, **params):
        rates = self.rate_tensor_func(value, **params).float()
        return torch.log(rates)

    def eval_log_on_geom(self, **params):
        return self.log_rate(
            self.coord_geom.shaped_grid, 
            **params
            )


    def _exp_log_rate(self, *args, **kwargs):
        return torch.exp(self.log_rate(*args, **kwargs))
