"""
REPLACES the gammabayes module defining SingleDMSpectralComp (the file loading PPPC_Tables).
Requires cosmixs_reader.py in the SAME directory as PPPC_reader.py, and the CosmiXs table at
<data_path>/CosmiXs_Tables/AtProduction-Gamma_COSMIXs.dat (github.com/ajueid/CosmiXs);
for the delta band: qcdunc_reader.py + <data_path>/QCDUNC_Tables/AtProduction-Ga.dat
(or the same table named AtProduction-Ga-Combined-Up-QCDUNC.dat)
(the data/wUncertainty file from github.com/ajueid/qcd-dm.github.io).

reader kwarg selects the CENTRAL table ('cosmixs' | 'pppc'); the OTHER becomes the kappa
blend partner:  log dNdE(kappa) = (1-kappa)*log central + kappa*log alt.
kappa=None (default) -> pure central, back-compatible.
"""
import os, torch
from typing import Literal
from functools import partial
from gammabayes import RegularTorchInterpolator
from gammabayes.core.batching import as_param
from gammabayes.priors.spectral_components import BaseSpectral_PriorComp
single_channel_spectral_data_path = os.path.dirname(os.path.dirname(__file__))
from .PPPC_reader import PPPCReader
from .cosmixs_reader import CosmiXsReader

_TABLES = {
    'cosmixs': (CosmiXsReader, "/CosmiXs_Tables/AtProduction-Gamma_COSMIXs.dat",
                CosmiXsReader.darkSUSY_to_CosmiXs_converter),
    'pppc':    (PPPCReader, "/PPPC_Tables/AtProduction_gamma_EW_corrections.dat",
                PPPCReader.darkSUSY_to_PPPC_converter),
}


# QCDUNC wUncertainty table (github.com/ajueid/qcd-dm.github.io, data/wUncertainty/AtProduction-Ga.dat);
# same file under either name.  First existing one wins.
_QCDUNC_NAMES = ("AtProduction-Ga.dat", "AtProduction-Ga-Combined-Up-QCDUNC.dat")


def _qcdunc_table_path():
    d = os.path.join(single_channel_spectral_data_path, "QCDUNC_Tables")
    for n in _QCDUNC_NAMES:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {_QCDUNC_NAMES} in {d}")


_WARNED = set()


def _warn_once(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"*** WARNING [SingleDMSpectralComp] {msg}", flush=True)


class SingleDMSpectralComp(BaseSpectral_PriorComp):
    """Single-channel DM spectra; reader-selectable central table + kappa blend to the other."""

    @staticmethod
    def _load_channel_from_reader(rd, channel, conv=None):
        try:
            grid = rd[(conv or {}).get(channel, channel)].reshape(rd.output_shape)
        except Exception:
            grid = rd[channel].reshape(rd.output_shape)
        return RegularTorchInterpolator(
            (torch.log10(torch.tensor(rd.mass_axis)), torch.tensor(rd.log10x_axis)),
            torch.sqrt(torch.tensor(grid)))          # sqrt-interp preserves positivity

    @classmethod
    def _load_channel(cls, name, channel):
        reader_cls, table, conv = _TABLES[name]
        return cls._load_channel_from_reader(reader_cls(single_channel_spectral_data_path + table), channel, conv)

    def __init__(self, binning_geometry=None, channel='W+W-', mass=torch.tensor(1.),
                 reader: Literal['cosmixs', 'pppc'] = 'cosmixs', *args, **kwargs):
        self.channel = channel
        self.reader = str(reader).lower()
        if self.reader not in _TABLES:
            raise ValueError(f"reader must be one of {list(_TABLES)}, got {reader!r}")
        _alt = 'pppc' if self.reader == 'cosmixs' else 'cosmixs'
        try:
            self._interp_central = self._load_channel(self.reader, channel)
        except Exception as e:
            print(f"SingleDMSpectralComp: {self.reader} table unavailable ({e}); using {_alt}")
            self.reader, _alt = _alt, self.reader
            self._interp_central = self._load_channel(self.reader, channel)
        try:
            self._interp_alt = self._load_channel(_alt, channel)   # kappa blend partner
        except Exception as e:
            self._interp_alt = None
            _warn_once(f"kappa blend partner table ({_alt}) unavailable for {channel}: {e!r} -> kappa has NO effect")

        # QCDUNC +1sigma FRACTIONAL band (Had (+) Scale (+) cNS in quadrature), applied to the
        # CosmiXs central as a ratio: dN_up = dN_central * (up/central)_QCDUNC. Transferring the
        # RELATIVE band avoids mixing the QCDUNC<->CosmiXs code difference (that's kappa's job).
        self._interp_band_ratio = None
        try:
            import numpy as _np
            from .qcdunc_reader import QCDUNCReader
            _band = QCDUNCReader(_qcdunc_table_path())
            _q = QCDUNCReader.darkSUSY_to_QCDUNC_converter[channel]
            _cen = _band[_q]
            _ratio = _np.where(_cen > 0, _band[_q + '_up'] / _np.maximum(_cen, 1e-300), 1.0)
            _ratio = _np.clip(_ratio, 1.0, 1e3).reshape(_band.output_shape)   # up-edge >= central
            self._interp_band_ratio = RegularTorchInterpolator(
                (torch.log10(torch.tensor(_band.mass_axis)), torch.tensor(_band.log10x_axis)),
                torch.sqrt(torch.tensor(_ratio)))
        except Exception as e:
            self._interp_band_ratio = None
            _warn_once(f"QCDUNC band table ({' or '.join(_QCDUNC_NAMES)} in <dark_matter>/QCDUNC_Tables/) unusable "
                       f"for {channel}: {e!r} -> delta has NO effect on the spectrum")

        self.loglog10 = torch.log(torch.log(torch.tensor(10.)))
        self.log_spectral_gen = self.__log_spectral_gen
        super().__init__(logfunc=self.log_spectral_gen)

    def _log_dNdE_one(self, interp, energy, log10mass):
        """energy (1, E), log10mass (B, 1) -> (B, E)."""
        spec = (interp((log10mass, torch.log10(energy) - log10mass))) ** 2
        # clamp before log keeps the kappa blend finite where one table is exactly 0
        return torch.log(spec.clamp(min=1e-37)) - torch.log(energy) - self.loglog10

    def __log_spectral_gen(self, energy, mass, kappa=None, delta=None):
        """Batched log dN/dE.

        energy: (E,) axis.  mass, kappa, delta: scalar or (B,).  Returns (B, E).
        """
        E = torch.as_tensor(energy).reshape(1, -1)
        mass = as_param(mass, 1, like=E)
        log10mass = torch.log10(mass)
        log_spec = self._log_dNdE_one(self._interp_central, E, log10mass)
        if (kappa is not None) and (self._interp_alt is not None):
            kappa = as_param(kappa, 1, like=E)
            log_alt = self._log_dNdE_one(self._interp_alt, E, log10mass)
            log_spec = (1 - kappa) * log_spec + kappa * log_alt      # code choice: CosmiXs <-> PPPC
        if (delta is not None) and (self._interp_band_ratio is not None):
            delta = as_param(delta, 1, like=E)
            sqrt_ratio = self._interp_band_ratio((log10mass, torch.log10(E) - log10mass))
            log_spec = log_spec + delta * 2.0 * torch.log(sqrt_ratio.clamp(min=1.0))
        return torch.where(E > mass, -50 - 3 * torch.log(E), log_spec)
