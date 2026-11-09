"""A1 batching tests.  Run:  pytest gammabayes/tests/test_batching.py -q

Checks, for every batched component:
  * B == 1 (scalar params) reproduces the pre-batching scalar formula,
  * B > 1 equals a python loop over B single evaluations (stacked).
No IRFs, PPPC tables or GALPROP data are needed: the DM spectral tables are
replaced by a synthetic interpolator via monkeypatch.
"""
import math
import pytest
import torch

from gammabayes.core.batching import as_param, batch_size, promote_batch
from gammabayes.core.reg_interpolator import RegularTorchInterpolator
from gammabayes.core.binning_geometry import GammaBinning
from gammabayes.priors.spectral_components import PowerLaw, ExpCutoffPowerLaw, BrokenPowerLaw, LogParabola
from gammabayes.priors.spatial_components import (PointSpatial_PriorComp, GaussianSpatial_PriorComp,
                                                  IsotropicSpatial_PriorComp)
from gammabayes.priors import TwoCompFluxPrior
from gammabayes.dark_matter.density_profiles import Einasto_Profile, GNFW_Profile
from gammabayes.dark_matter.PPPC_Tables import single_channel_DM_Spectral_Class as scdm
from gammabayes.dark_matter.custom_ratios_model import CustomDMRatiosModel

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
B = 5



@pytest.fixture(scope="module")
def geom():
    return GammaBinning(energy_res=4, energy_min=0.3, energy_max=30.,
                        lon_res=5, lon_min=-2., lon_max=2.,
                        lat_res=5, lat_min=-1.5, lat_max=1.5)


class _FakeExposure:
    """Stand-in for GammaLogExposure: log exposure = log(E) + 0.1*lon."""
    pointing_dirs = None
    live_times = None
    def __call__(self, e, lon, lat):
        return torch.log(e) + 0.1 * lon


def _loop(fn, params, B):
    """Evaluate fn with the i-th slice of every (B,) param, stack over i."""
    outs = []
    for i in range(B):
        p = {k: (v[i] if torch.is_tensor(v) and v.ndim == 1 else v) for k, v in params.items()}
        outs.append(fn(**p))
    return torch.stack(outs, 0)


def _squeeze1(x):
    return x[0] if x.shape[0] == 1 else x



def test_as_param_and_batch_size():
    assert as_param(2.0, 2).shape == (1, 1, 1)
    assert as_param(torch.tensor(2.0), 1).shape == (1, 1)
    assert as_param(torch.arange(4.), 2).shape == (4, 1, 1)
    with pytest.raises(ValueError):
        as_param(torch.zeros(2, 2), 1)
    assert batch_size(1., torch.tensor(2.), torch.zeros(3), None) == 3
    with pytest.raises(ValueError):
        batch_size(torch.zeros(3), torch.zeros(4))
    assert promote_batch(torch.zeros(4, 5), 2).shape == (1, 4, 5)
    assert promote_batch(torch.zeros(3, 4, 5), 2).shape == (3, 4, 5)



def test_interpolator_broadcast_matches_flat():
    ax0 = torch.linspace(0, 1, 7)
    ax1 = torch.linspace(-1, 1, 9)
    vals = torch.randn(7, 9)
    itp = RegularTorchInterpolator((ax0, ax1), vals)
    x0 = torch.rand(B)
    x1 = torch.rand(6) * 2 - 1
    out = itp((x0[:, None], x1[None, :]))
    assert out.shape == (B, 6)
    flat = itp((x0[:, None].expand(B, 6).reshape(-1), x1[None, :].expand(B, 6).reshape(-1)))
    assert torch.allclose(out, flat.reshape(B, 6))
    # exact on grid nodes
    assert torch.allclose(itp((ax0[2].reshape(1), ax1[3].reshape(1))), vals[2, 3].reshape(1))
    # edge clamping
    assert torch.allclose(itp((torch.tensor([5.]), torch.tensor([0.]))), itp((torch.tensor([1.]), torch.tensor([0.]))))



def test_power_laws(geom):
    E = geom.energy_axis
    idx = torch.rand(B) + 2.; phi0 = 10 ** (-13 + torch.rand(B))
    pl = PowerLaw()
    out = pl(E, index=idx, phi0=phi0)
    assert out.shape == (B, len(E))
    ref = torch.stack([torch.log(phi0[i]) - idx[i] * torch.log(E / 1.) for i in range(B)])
    assert torch.allclose(out, ref)
    # scalar call -> (1, E) and equals old scalar formula
    out1 = pl(E, index=idx[0], phi0=phi0[0])
    assert out1.shape == (1, len(E)) and torch.allclose(out1[0], ref[0])

    for cls, kw in [(ExpCutoffPowerLaw, dict(index=idx, phi0=phi0, lambdaval=torch.rand(B) / 50)),
                    (BrokenPowerLaw, dict(index1=idx, index2=idx + 1, phi0=phi0)),
                    (LogParabola, dict(index=idx, phi0=phi0, beta=torch.rand(B)))]:
        comp = cls()
        out = comp(E, **kw)
        loop = _loop(lambda **p: comp(E, **p)[0], kw, B)
        assert out.shape == (B, len(E)) and torch.allclose(out, loop), cls.__name__



def test_point_and_gaussian_spatial(geom):
    ll = geom.shaped_grid[0, ..., 1:]
    lon = torch.rand(B) * 2 - 1; lat = torch.rand(B) - 0.5
    pt = PointSpatial_PriorComp()
    out = pt(ll, pos_lon=lon, pos_lat=lat)
    assert out.shape == (B, *ll.shape[:-1])
    loop = _loop(lambda **p: pt(ll, **p)[0], dict(pos_lon=lon, pos_lat=lat), B)
    assert torch.allclose(out, loop)
    # tent deposition integrates to 1 (per deg^2 normalisation)
    dlon = geom.lon_axis[1] - geom.lon_axis[0]; dlat = geom.lat_axis[1] - geom.lat_axis[0]
    assert torch.allclose((out.exp() * dlon * dlat).sum((1, 2)), torch.ones(B), atol=1e-6)

    g = GaussianSpatial_PriorComp()
    kw = dict(pos_lon=lon, pos_lat=lat, sigma_lon=torch.rand(B) * .3 + .1, sigma_lat=torch.rand(B) * .3 + .1, rho=torch.rand(B) * .5)
    out = g(ll, **kw)
    loop = _loop(lambda **p: g(ll, **p)[0], kw, B)
    assert out.shape == (B, *ll.shape[:-1]) and torch.allclose(out, loop)

    iso = IsotropicSpatial_PriorComp()
    assert iso(ll).shape == (1, *ll.shape[:-1])



def _old_logdiffJ_einasto(prof, theta_deg, alpha):
    """Verbatim pre-batching LOS integral (scalar alpha) for the reference."""
    th = torch.deg2rad(theta_deg).clamp(min=torch.deg2rad(torch.tensor(prof.theta_min_deg)))
    D = prof.DISTANCE
    b = D * torch.sin(th); s0 = D * torch.cos(th); s_max = prof.t_range[-1] * D
    u_lo = torch.atan(-s0 / b); u_hi = torch.atan((s_max - s0) / b)
    N = prof.int_resolution
    frac = (torch.arange(N, dtype=th.dtype) + 0.5) / N
    u = u_lo[:, None] + (u_hi - u_lo)[:, None] * frac[None, :]
    log_du = torch.log((u_hi - u_lo) / N)[:, None]
    cos_u = torch.cos(u); r = b[:, None] / cos_u
    log_norm_corr = torch.log(prof.LOCAL_DENSITY) - prof.log_profile_func(D, alpha=alpha)
    log_rho_pow = 2. * (prof.log_profile_func(r, alpha=alpha) + log_norm_corr)
    log_ds = torch.log(b)[:, None] - 2. * torch.log(cos_u) + log_du
    return torch.logsumexp(log_rho_pow + log_ds, dim=1) + torch.log(prof.d_kpc_to_cm * prof.sr_to_deg2)


def test_dm_profiles(geom):
    ll = geom.shaped_grid[0, ..., 1:]
    theta = torch.linspace(0, 2.5, 40)
    ein = Einasto_Profile()
    alpha = torch.rand(B) * 0.3 + 0.1
    out = ein._log_diffJ_from_offset(theta, {"alpha": alpha})
    assert out.shape == (B, 40)
    for i in range(B):
        assert torch.allclose(out[i], _old_logdiffJ_einasto(ein, theta, alpha[i]), rtol=1e-10)
    # full-grid call and scalar call
    full = ein(ll, alpha=alpha)
    assert full.shape == (B, *ll.shape[:-1])
    one = ein(ll, alpha=alpha[2])
    assert one.shape == (1, *ll.shape[:-1]) and torch.allclose(one[0], full[2])
    # chunking path gives identical numbers
    ein_small = Einasto_Profile(max_elems=40 * ein.int_resolution * 2)   # chunk of 2
    assert torch.allclose(ein_small._log_diffJ_from_offset(theta, {"alpha": alpha}), out)
    # gNFW with free gamma and r_s batched
    g = GNFW_Profile()
    kw = dict(gamma=torch.rand(B) * .8 + .5, r_s=torch.rand(B) * 10 + 20)
    out = g(ll, **kw)
    loop = _loop(lambda **p: g(ll, **p)[0], kw, B)
    assert torch.allclose(out, loop)


# ------------------------------------------------- DM spectral + ratios
@pytest.fixture
def fake_tables(monkeypatch):
    """Replace table loading with a smooth synthetic dN/dx so no data is needed."""
    m_axis = torch.logspace(-1, 2, 25)                 # TeV
    x_axis = torch.linspace(-6, 0, 60)                 # log10(E/m)
    def mk(scale):
        lm, lx = torch.meshgrid(torch.log10(m_axis), x_axis, indexing="ij")
        grid = scale * torch.exp(-0.5 * ((lx + 1.5) / 0.8) ** 2) * (1 + 0.1 * lm)
        return RegularTorchInterpolator((torch.log10(m_axis), x_axis), torch.sqrt(grid))
    def fake_load(cls, name, channel):
        return mk(1.0 if name == "cosmixs" else 1.3)
    monkeypatch.setattr(scdm.SingleDMSpectralComp, "_load_channel", classmethod(fake_load))
    return mk


def test_single_dm_spectral(geom, fake_tables):
    comp = scdm.SingleDMSpectralComp(binning_geometry=geom, channel="W+W-")
    comp._interp_band_ratio = fake_tables(2.0)      # synthetic +1sigma band ratio (>=1 after clamp)
    E = geom.energy_axis
    mass = torch.rand(B) * 5 + 0.5; kappa = torch.rand(B); delta = torch.rand(B) * 2 - 1
    out = comp(E, mass=mass, kappa=kappa, delta=delta)
    assert out.shape == (B, len(E))
    loop = _loop(lambda **p: comp(E, **p)[0], dict(mass=mass, kappa=kappa, delta=delta), B)
    assert torch.allclose(out, loop)
    # above-mass floor applied per batch element
    hi = out[E[None, :].expand(B, -1) > mass[:, None]]
    assert torch.all(hi < -40)
    # old scalar formula at B = 1
    m0 = mass[0]
    lm = torch.log10(m0)
    ref = torch.log((comp._interp_central((lm.expand(len(E)), torch.log10(E) - lm)) ** 2).clamp(min=1e-37)) \
        - torch.log(E) - comp.loglog10
    ref = torch.where(E > m0, -50 - 3 * torch.log(E), ref)
    assert torch.allclose(comp(E, mass=m0)[0], ref)


def test_custom_ratios_model(geom, fake_tables):
    ch = ["W+W-", "ZZ", "tt"]
    model = CustomDMRatiosModel(binning_geometry=geom, channels=ch, log_exposure_map=_FakeExposure())
    E = geom.energy_axis
    mass = torch.rand(B) * 5 + 0.5
    r = torch.rand(B, 3); r = r / r.sum(1, keepdim=True)
    alpha = torch.rand(B) * 0.3 + 0.1
    spec = {"mass": mass, "sigmav": 1e-26, "ratios": r, "kappa": torch.rand(B)}
    spat = {"alpha": alpha}
    out = model.eval_log_on_geom(spectral_params=spec, spatial_params=spat)
    assert out.shape == (B, *geom.shape)
    # dict ratios interface agrees with tensor interface
    rd = {k: r[:, j] for j, k in enumerate(ch)}
    out_d = model.eval_log_on_geom(spectral_params={**spec, "ratios": rd}, spatial_params=spat)
    assert torch.allclose(out, out_d)
    # equals loop of single evaluations (cache must not leak between calls)
    for i in range(B):
        si = {"mass": mass[i], "sigmav": 1e-26, "ratios": r[i], "kappa": spec["kappa"][i]}
        one = model.eval_log_on_geom(spectral_params=si, spatial_params={"alpha": alpha[i]})
        assert one.shape == geom.shape                 # squeezed at B == 1
        assert torch.allclose(one, out[i])
    # squeeze=False keeps the axis
    assert model.eval_log_on_geom(spectral_params=si, spatial_params={"alpha": alpha[i]}, squeeze=False).shape == (1, *geom.shape)
    # separable: full cube == spectrum + J + exposure
    lS = model.log_spectrum(mass=mass, sigmav=1e-26, ratios=r, kappa=spec["kappa"])
    lJ = model.log_J(alpha=alpha)
    g = geom.shaped_grid
    lX = _FakeExposure()(g[..., 0], g[..., 1], g[..., 2])
    assert torch.allclose(out, lS[:, :, None, None] + lJ[:, None] + lX[None])
    # spectral cache hit: moving only alpha must not change the spectrum object
    s1 = model.log_spectrum(mass=mass, sigmav=1e-26, ratios=r, kappa=spec["kappa"])
    assert s1 is lS
    model.log_J(alpha=alpha + 0.01)
    assert model.log_spectrum(mass=mass, sigmav=1e-26, ratios=r, kappa=spec["kappa"]) is lS

########################################################################
#  TwoCompFluxPrior
########################################################################


def test_two_comp_broadcast(geom):
    prior = TwoCompFluxPrior(spectral_comp=PowerLaw(), spatial_comp=PointSpatial_PriorComp(),
                             binning_geometry=geom, log_exposure_map=_FakeExposure())
    idx = torch.rand(B) + 2
    # spectral batched, spatial scalar -> (B, E, lon, lat)
    out = prior.eval_log_on_geom(spectral_params={"index": idx}, spatial_params={"pos_lon": 0.3, "pos_lat": -0.2})
    assert out.shape == (B, *geom.shape)
    # spatial batched, spectral scalar
    out2 = prior.eval_log_on_geom(spectral_params={"index": 2.4}, spatial_params={"pos_lon": torch.rand(B), "pos_lat": -0.2})
    assert out2.shape == (B, *geom.shape)
    # mismatched batch sizes raise
    with pytest.raises(RuntimeError):
        prior.eval_log_on_geom(spectral_params={"index": idx}, spatial_params={"pos_lon": torch.rand(B + 1), "pos_lat": 0.})
    # scalar everywhere -> squeezed
    assert prior.eval_log_on_geom(spectral_params={"index": 2.4}, spatial_params={"pos_lon": 0., "pos_lat": 0.}).shape == geom.shape


def test_dm_profile_checkpoint_gradient(geom):
    ll = geom.shaped_grid[0, ..., 1:]
    alpha = (torch.rand(B) * 0.3 + 0.1).requires_grad_(True)
    S = ll.shape[0] * ll.shape[1]
    ck = Einasto_Profile(max_elems=S * 101 * 2, grad_checkpoint=True)       # chunks of 2, checkpointed
    nock = Einasto_Profile(max_elems=S * 101 * 2, grad_checkpoint=False)
    g1, = torch.autograd.grad(ck(ll, alpha=alpha).sum(), alpha)
    g2, = torch.autograd.grad(nock(ll, alpha=alpha).sum(), alpha)
    assert torch.allclose(g1, g2, rtol=1e-10)
    # cache is bypassed under grad: output must carry a grad_fn
    assert ck(ll, alpha=alpha).requires_grad


def test_custom_ratios_large_batch_no_grad(geom, fake_tables):
    """Regression: B > cache_max_batch under no_grad must bypass the cache, not raise."""
    model = CustomDMRatiosModel(binning_geometry=geom, channels=["W+W-", "ZZ"], log_exposure_map=_FakeExposure())
    Bb = 40
    r = torch.rand(Bb, 2); r = r / r.sum(1, keepdim=True)
    with torch.no_grad():
        out = model.eval_log_on_geom(spectral_params={"mass": torch.rand(Bb) * 5 + 0.5, "sigmav": 1e-26, "ratios": r},
                                     spatial_params={"alpha": torch.rand(Bb) * 0.3 + 0.1})
    assert out.shape == (Bb, *geom.shape)
    assert model._spec_cache is None or model._spec_cache[1].shape[0] <= model.cache_max_batch
