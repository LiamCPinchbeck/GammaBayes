"""A3 tests: energy-calibrated PSF.  Run:  pytest gammabayes/tests/test_psf_energy.py -q
Uses a fake gammapy-like PSF (2-D Gaussian with sigma(E, offset)) so no IRF files are needed.
"""
import numpy as np
import pytest
import torch
from astropy import units as u

from gammabayes.core.binning_geometry import GammaBinning
from gammabayes.response_functions.core.irf_extractor_class import IRFExtractor
from gammabayes.response_functions.core.irf_kernel import (make_spatial_kernel_per_energy,
                                                           make_irf_kernel_per_energy, make_irf_kernel)
from gammabayes.utils.log_space_conv_class import LogSumExpConv3D

torch.set_default_dtype(torch.float64)
R68_OVER_SIGMA = np.sqrt(-2 * np.log(1 - 0.68))      # 1.5096


class _FakePSF:
    """sigma(E, off) = 0.1 deg * (E/1TeV)^-0.3 * (1 + 0.1*off)."""
    @staticmethod
    def sigma(E_TeV, off_deg):
        return 0.1 * E_TeV ** -0.3 * (1 + 0.1 * off_deg)

    def evaluate(self, energy_true, rad, offset):
        E = energy_true.to(u.TeV).value; r = rad.to(u.deg).value; o = offset.to(u.deg).value
        s = self.sigma(E, o)
        s_rad = np.deg2rad(s)
        val = np.exp(-0.5 * (np.deg2rad(r) / s_rad) ** 2) / (2 * np.pi * s_rad ** 2)
        return val / u.sr


class _FakeAeff:
    def evaluate(self, energy_true, offset):
        E = energy_true.to(u.TeV).value; o = offset.to(u.deg).value
        return (1e6 * E ** 0.5 * np.exp(-o / 3.0)) * u.cm ** 2


@pytest.fixture
def irfs():
    ex = object.__new__(IRFExtractor)
    ex.psf_default = _FakePSF()
    ex.aeff_default = _FakeAeff()
    ex.psf_units = (1 / u.sr).unit
    ex.aeff_units = u.cm ** 2
    return ex


@pytest.fixture
def geom():
    return GammaBinning(energy_res=4, energy_min=0.3, energy_max=30.,
                        lon_res=10, lon_min=-2., lon_max=2., lat_res=10, lat_min=-2., lat_max=2.)


def test_containment_radius(irfs):
    E = np.array([0.5, 2., 10.]); off = np.array([0., 1.5])
    R = irfs.psf_containment_radius(E, off, fraction=0.68)
    assert R.shape == (3, 2)
    assert np.allclose(R, R68_OVER_SIGMA * _FakePSF.sigma(E[:, None], off[None, :]), rtol=3e-3)


def test_sigma_vs_energy_uniform_and_aeff(irfs, geom):
    sig_u = irfs.psf_sigma_vs_energy(geom, weighting="uniform", pixel_floor=False, pointing_dir=torch.tensor([0., 0.]))
    assert sig_u.shape == (len(geom.energy_axis),)
    E = geom.energy_axis.numpy()
    lon, lat = torch.meshgrid(geom.lon_axis, geom.lat_axis, indexing="ij")
    off = torch.sqrt(lon ** 2 + lat ** 2).flatten().numpy()           # small-angle ok at 2 deg to <0.1%
    expect = _FakePSF.sigma(E[:, None], off[None, :]).mean(1)
    assert np.allclose(sig_u.numpy(), expect, rtol=5e-3)
    # aeff weighting prefers small offsets -> smaller sigma than uniform
    sig_a = irfs.psf_sigma_vs_energy(geom, weighting="aeff", pixel_floor=False, pointing_dir=torch.tensor([0., 0.]))
    assert np.all(sig_a.numpy() < sig_u.numpy())
    # counts weighting with a flat cube == uniform
    sig_c = irfs.psf_sigma_vs_energy(geom, weighting="counts", counts=torch.ones(geom.shape), pixel_floor=False,
                                     pointing_dir=torch.tensor([0., 0.]))
    assert np.allclose(sig_c.numpy(), sig_u.numpy())
    # pixel floor adds Delta^2/12 in quadrature
    sig_f = irfs.psf_sigma_vs_energy(geom, weighting="uniform", pixel_floor=True, pointing_dir=torch.tensor([0., 0.]))
    d = float(geom.lon_axis[1] - geom.lon_axis[0])
    assert np.allclose(sig_f.numpy() ** 2, sig_u.numpy() ** 2 + d ** 2 / 12)
    # monotone: sigma falls with energy for this PSF
    assert np.all(np.diff(sig_u.numpy()) < 0)


def test_per_energy_kernel_builder(geom):
    E = len(geom.energy_axis)
    sigma = torch.linspace(0.3, 0.05, E)
    k = make_spatial_kernel_per_energy(geom, sigma, confidence_clearance=4)
    assert k.ndim == 3 and k.shape[0] == E and k.shape[1] % 2 == 1 and k.shape[2] % 2 == 1
    assert torch.allclose(k.sum((1, 2)), torch.ones(E))
    # width ordering: second moment decreases with energy
    ax = (torch.arange(k.shape[1]) - k.shape[1] // 2) * (geom.lon_axis[1] - geom.lon_axis[0])
    var = (k.sum(2) * ax[None] ** 2).sum(1)
    assert torch.all(torch.diff(var) < 0)
    # constant sigma -> all planes identical
    kc = make_spatial_kernel_per_energy(geom, torch.full((E,), 0.1))
    assert torch.allclose(kc, kc[:1].expand(E, -1, -1))
    # plugs straight into the conv and runs
    kE, kS = make_irf_kernel_per_energy(geom, torch.tensor(0.09), sigma, confidence_clearance=2.)
    conv = LogSumExpConv3D(energy_kernel=kE, spatial_kernel=kS, binning_geometry=geom, padding=None)
    x = torch.randn(2, *geom.shape)
    assert conv(x).shape == x.shape
    # energy kernel identical to the legacy builder's energy factor
    legacy = make_irf_kernel(geom, torch.tensor(0.09), torch.tensor(0.1), torch.tensor(0.1), confidence_clearance=2.)
    assert torch.allclose(kE, legacy.sum((1, 2)) / legacy.sum() * kE.sum())
