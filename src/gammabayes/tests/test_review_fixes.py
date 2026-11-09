"""Regression tests for the 2.1.1 review fixes.  Run:  pytest gammabayes/tests/test_review_fixes.py -q
No IRF files, DM tables or GALPROP data are needed."""
import numpy as np
import pytest
import torch
from astropy import units as u

from gammabayes.core.binning_geometry import GammaBinning
from gammabayes.core.exposure import GammaLogExposure
from gammabayes.core.reg_interpolator import RegularTorchInterpolator
from gammabayes.priors import TwoCompFluxPrior, StaticSourceDistTemplate
from gammabayes.priors.astro_sources import IsotropicPowerLaw, IsotropicBrokenPowerLaw
from gammabayes.priors.spectral_components import PowerLaw
from gammabayes.priors.spatial_components import IsotropicSpatial_PriorComp
from gammabayes.response_functions.core.irf_kernel import IRFKernelDist
from gammabayes.utils.integration import construct_log_dx, logspace_riemann

torch.set_default_dtype(torch.float64)


class _FakeIRFs:
    """IRFExtractor stand-in: flat A_eff of 1e6 cm^2 inside 3 deg, NaN outside."""
    aeff_units = u.cm ** 2
    pointing_dir = torch.tensor([0.5, -0.5])

    def log_aeff(self, energy, lon, lat, pointing_dir=None, parameters=None):
        off = torch.hypot(lon - pointing_dir[0], lat - pointing_dir[1])
        out = torch.full_like(energy, float(np.log(1e6)))
        return torch.where(off < 3.0, out, torch.full_like(out, float("nan")))


@pytest.fixture
def geom():
    return GammaBinning(energy_res=4, energy_min=0.3, energy_max=30.,
                        lon_res=4, lon_min=-2., lon_max=2., lat_res=4, lat_min=-2., lat_max=2.)


# ------------------------------------------------------------------ geometry
def test_geometry_equality_and_padding(geom):
    other = GammaBinning(energy_res=4, energy_min=0.3, energy_max=30.,
                         lon_res=4, lon_min=-2., lon_max=2., lat_res=4, lat_min=-2., lat_max=2.)
    assert geom == other and not (geom != other)
    assert geom != GammaBinning(energy_res=4, energy_min=0.3, energy_max=30., lon_res=4, lon_min=-1., lon_max=1., lat_res=4, lat_min=-2., lat_max=2.)
    pad = geom.padded((2, 1, 3))
    assert tuple(len(a) for a in pad.axes) == tuple(n + 2 * p for n, p in zip(geom.shape, (2, 1, 3)))
    for a_t, a_r, p in zip(pad.axes, geom.axes, (2, 1, 3)):
        assert torch.allclose(a_t[p:p + len(a_r)], a_r)
    assert "GammaBinning" in repr(geom)


# ------------------------------------------------------------------ exposure
def test_exposure_default_pointing_and_none_livetime(geom):
    # pointing_dirs=None used to raise TypeError (torch.stack([None])); live_times=None raised too
    ex = GammaLogExposure(binning_geometry=geom, irfs=_FakeIRFs(), pointing_dirs=None, live_times=None)
    assert ex.pointing_dirs.shape == (1, 2) and torch.allclose(ex.pointing_dirs[0], _FakeIRFs.pointing_dir)
    assert ex.live_times.shape == (1,) and float(ex.live_times[0]) == 1.0
    assert torch.isfinite(ex.log_exposure_map).any() and torch.isneginf(ex.log_exposure_map).any()   # NaN -> -inf


def test_exposure_livetime_units_consistent(geom):
    """1 h given as hours or as 3600 s must give the same map, and __call__ must agree with the map."""
    ex_h = GammaLogExposure(binning_geometry=geom, irfs=_FakeIRFs(), pointing_dirs=[[0., 0.]], live_times=1 * u.hr)
    ex_s = GammaLogExposure(binning_geometry=geom, irfs=_FakeIRFs(), pointing_dirs=[[0., 0.]], live_times=3600 * u.s)
    assert torch.allclose(ex_h.log_exposure_map, ex_s.log_exposure_map)
    # the map is A_eff * t * bin volume:  1e6 cm2 * 3600 s * dE dlon dlat
    i = (len(geom.energy_axis) // 2, len(geom.lon_axis) // 2, len(geom.lat_axis) // 2)
    expected = np.log(1e6 * 3600) + float(torch.log(geom.bin_width_mat[i]))
    assert abs(float(ex_s.log_exposure_map[i]) - expected) < 1e-9
    g = geom.shaped_grid
    called = ex_s(g[..., 0].flatten(), g[..., 1].flatten(), g[..., 2].flatten()).reshape(geom.shape)
    fin = torch.isfinite(ex_s.log_exposure_map)
    assert torch.allclose(called[fin], ex_s.log_exposure_map[fin])          # no double unit factor
    # zero exposure comes back as a finite floor, never -inf (NaN-safe gradients through logaddexp)
    assert torch.isfinite(called).all() and (called[~fin] < -600).all()


def test_exposure_addition(geom):
    ex = GammaLogExposure(binning_geometry=geom, irfs=_FakeIRFs(), pointing_dirs=[[0., 0.]], live_times=10 * u.s)
    tot = ex + ex
    fin = torch.isfinite(ex.log_exposure_map)
    assert torch.allclose(tot.log_exposure_map[fin], ex.log_exposure_map[fin] + np.log(2.))
    assert tot.pointing_dirs.shape == (2, 2) and tot.live_times.shape == (2,)


# -------------------------------------------------------------- interpolator
def test_interpolator_device_cache_hit():
    itp = RegularTorchInterpolator((torch.linspace(0, 1, 5),), torch.arange(5.))
    assert float(itp((torch.tensor(0.5),))) == pytest.approx(2.0)
    assert len(itp._dev_cache) == 1
    itp((torch.tensor([0.1, 0.9]),))
    assert len(itp._dev_cache) == 1


# ------------------------------------------------------------------- priors
class _FakeExposure:
    pointing_dirs = None
    live_times = None
    def __call__(self, e, lon, lat):
        return torch.zeros_like(e)


def test_isotropic_priors_instantiate_and_batch(geom):
    iso = IsotropicPowerLaw(binning_geometry=geom, index=2.5, phi0=1e-1, log_exposure_map=_FakeExposure())
    out = iso.eval_log_on_geom(spectral_params={"index": torch.tensor([2.0, 3.0])})
    assert out.shape == (2, *geom.shape)
    # uniform on the sky
    assert torch.allclose(out[0, :, 0, 0], out[0, :, -1, -1])
    ref = PowerLaw(index=2.0, phi0=1e-1)(geom.energy_axis)[0]
    assert torch.allclose(out[0, :, 0, 0], ref)
    bpl = IsotropicBrokenPowerLaw(binning_geometry=geom, log_exposure_map=_FakeExposure())
    assert bpl.eval_log_on_geom().shape == geom.shape


def test_static_source_template_both_inputs(geom):
    T = torch.rand(*geom.shape) + 0.1
    a = StaticSourceDistTemplate(source_flux_tensor_template=T, binning_geometry=geom, log_exposure_map=_FakeExposure())
    b = StaticSourceDistTemplate(source_flux_tensor_logtemplate=T.log(), binning_geometry=geom, log_exposure_map=_FakeExposure())
    assert torch.allclose(a.eval_log_on_geom(), T.log()) and torch.allclose(b.eval_log_on_geom(), T.log())
    assert b.func_template() is not None and torch.allclose(b.func_template(), T)


# --------------------------------------------------------------- integration
def test_construct_log_dx_linear_and_log():
    lin = torch.linspace(0., 1., 11)
    assert torch.allclose(construct_log_dx(lin).exp(), torch.full((11,), 0.1))
    log = torch.logspace(-1, 2, 7)
    dx = construct_log_dx(log).exp()
    assert torch.allclose(dx[:-1], torch.diff(log))
    # int_0.1^100 of 1 dx ~ Riemann sum of the widths
    assert abs(float(logspace_riemann(torch.zeros(7), log).exp()) - float(dx.sum())) < 1e-9


# ------------------------------------------------------------ kernel sampler
def test_vectorised_kernel_sampler_conserves_interior_counts(geom):
    torch.manual_seed(0)
    kd = IRFKernelDist(geom, irf_extractor=None, log_energy_sigma=torch.tensor(0.05),
                       lon_sigma=torch.tensor(0.1), lat_sigma=torch.tensor(0.1), kernel_sigmas=3)
    cube = torch.zeros(geom.shape)
    cube[len(geom.energy_axis) // 2, len(geom.lon_axis) // 2, len(geom.lat_axis) // 2] = 1000   # central voxel
    out = kd.sample_kernel(cube)
    assert out.shape == cube.shape
    assert float(out.sum()) == 1000.0            # nothing leaves the grid from the centre
    edge = torch.zeros(geom.shape); edge[0, 0, 0] = 1000
    assert float(kd.sample_kernel(edge).sum()) < 1000.0      # corner events migrate out
