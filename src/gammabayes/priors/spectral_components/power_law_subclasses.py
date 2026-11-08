from .base_spectral_comp import BaseSpectral_PriorComp
from gammabayes.core.batching import as_param
import torch
from functools import partial


def _E(energy):
    """energy (E,) -> (1, E) so (B, 1) parameters broadcast to (B, E)."""
    return torch.as_tensor(energy).reshape(1, -1)


class PowerLaw(BaseSpectral_PriorComp):

    @staticmethod
    def log_power_law(energy, ref_energy, index, phi0):
        E = _E(energy)
        index, phi0, ref_energy = (as_param(v, 1, like=E) for v in (index, phi0, ref_energy))
        return torch.log(phi0) - index * torch.log(E / ref_energy)

    def __init__(self, ref_energy=torch.tensor(1.), index=torch.tensor(2.5), phi0=torch.tensor(1e-13), *args, **kwargs):
        self.__logfunc = partial(self.log_power_law, ref_energy=ref_energy, index=index, phi0=phi0)
        super().__init__(logfunc=self.__logfunc)


class ExpCutoffPowerLaw(BaseSpectral_PriorComp):

    @staticmethod
    def log_exp_cutoff_power_law(energy, index, ref_energy, lambdaval, phi0):
        E = _E(energy)
        index, phi0, ref_energy, lambdaval = (as_param(v, 1, like=E) for v in (index, phi0, ref_energy, lambdaval))
        return torch.log(phi0) - index * torch.log(E / ref_energy) - lambdaval * E

    def __init__(self,
                 index=torch.tensor(2.5),
                 ref_energy=torch.tensor(1.), lambdaval=torch.tensor(1/100.),  # cutoff energy of 100 TeV
                 phi0=torch.tensor(1e-13), *args, **kwargs):
        self.__log_exp_cutoff_power_law = partial(self.log_exp_cutoff_power_law, ref_energy=ref_energy,
                                                  index=index, phi0=phi0, lambdaval=lambdaval)
        super().__init__(logfunc=self.__log_exp_cutoff_power_law)


class BrokenPowerLaw(BaseSpectral_PriorComp):

    @staticmethod
    def log_broken_power_law(energy, index1, index2, break_energy, phi0):
        E = _E(energy)
        index1, index2, break_energy, phi0 = (as_param(v, 1, like=E) for v in (index1, index2, break_energy, phi0))
        log_value1 = torch.log(phi0) - index1 * torch.log(E / break_energy)
        log_value2 = torch.log(phi0) - index2 * torch.log(E / break_energy)
        return torch.where(E > break_energy, log_value2, log_value1)

    def __init__(self,
                 index1=torch.tensor(2.), index2=torch.tensor(3.),
                 break_energy=torch.tensor(10.),
                 phi0=torch.tensor(1e-13), *args, **kwargs):
        self.__log_broken_power_law = partial(self.log_broken_power_law, index1=index1, index2=index2,
                                              phi0=phi0, break_energy=break_energy)
        super().__init__(logfunc=self.__log_broken_power_law)


class BroadBrokenPowerLaw(BaseSpectral_PriorComp):

    # Influenced by model in https://arxiv.org/pdf/astro-ph/0607333
    @staticmethod
    def log_broad_broken_power_law(energy, ref_energy, cutoff_energy_TeV,
                                   index1, index2, S, phi0):
        E = _E(energy)
        cutoff_energy_TeV, index1, index2, S, phi0 = (as_param(v, 1, like=E)
                                                      for v in (cutoff_energy_TeV, index1, index2, S, phi0))
        x = E / cutoff_energy_TeV
        # was: undefined `index` and `energy.value`; use the low-energy index as the base slope
        return torch.log(phi0) - index1 * torch.log(x) + S * (index1 - index2) * torch.log(1 + x ** (1 / S))

    def __init__(self, ref_energy=torch.tensor(1.), cutoff_energy_TeV=torch.tensor(10.),
                 index1=torch.tensor(2.), index2=torch.tensor(3.),
                 S=torch.tensor(0.3), phi0=torch.tensor(1e-12),
                 *args, **kwargs):
        self.__log_broad_broken_power_law = partial(self.log_broad_broken_power_law,
                                                    ref_energy=ref_energy, cutoff_energy_TeV=cutoff_energy_TeV,
                                                    index1=index1, index2=index2, S=S, phi0=phi0)
        super().__init__(logfunc=self.__log_broad_broken_power_law)
