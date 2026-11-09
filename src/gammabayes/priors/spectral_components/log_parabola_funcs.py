from .base_spectral_comp import BaseSpectral_PriorComp
from gammabayes.core.batching import as_param
import torch
from functools import partial


# Values from here https://arxiv.org/pdf/1903.06621

class LogParabola(BaseSpectral_PriorComp):
    """Log parabola phi0 (E / ref_energy)^-(index + beta log10(E / ref_energy)).

    The curvature term uses log10 while the power is taken with the natural logarithm, so ``beta`` is
    1 / ln(10) times the value in the gammapy / Fermi convention (which uses ln in both places).
    """

    @staticmethod
    def log_log_parabola_func(energy, index, phi0, beta, ref_energy):
        E = torch.as_tensor(energy).reshape(1, -1)
        index, phi0, beta, ref_energy = (as_param(v, 1, like=E) for v in (index, phi0, beta, ref_energy))
        energy_ratio = E / ref_energy
        exponent = -(index + beta * torch.log10(energy_ratio))
        return torch.log(phi0) + exponent * torch.log(energy_ratio)

    def __init__(self, index=torch.tensor(2.5), phi0=torch.tensor(1e-12), beta=torch.tensor(0.5),
                 ref_energy=torch.tensor(1.), *args, **kwargs):
        self.__log_log_parabola_func = partial(self.log_log_parabola_func,
                                               index=index, phi0=phi0, beta=beta,
                                               ref_energy=ref_energy)
        super().__init__(logfunc=self.__log_log_parabola_func)
