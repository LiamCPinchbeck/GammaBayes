from .base_spectral_comp import BaseSpectral_PriorComp
import torch
from functools import partial



# Values from here https://arxiv.org/pdf/1903.06621

class LogParabola(BaseSpectral_PriorComp):

    @staticmethod
    def log_log_parabola_func(self, energy, index, phi0, beta, ref_energy):

        energy_ratio = energy/ref_energy
        exponent = -(index + beta*np.log10(energy_ratio))
        log_value = np.log(phi0)+exponent*np.log(energy_ratio)

        return log_value

    def __init__(self, index=torch.tensor(2.5), phi0=torch.tensor(1e-12), beta=torch.tensor(0.5), ref_energy=torch.tensor(1.), *args, **kwargs):
            
        self.__log_log_parabola_func = partial(log_log_parabola_func, 
                                                index=index, phi0=phi0, beta=beta,
                                                ref_energy=ref_energy)

        super().__init__(logfunc=self.__log_log_parabola_func)