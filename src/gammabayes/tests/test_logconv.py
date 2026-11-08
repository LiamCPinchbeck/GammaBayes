"""A2 tests: separable LogSumExpConv3D == reference O(K^3) implementation.
Run:  pytest gammabayes/tests/test_logconv.py -q
"""
import pytest
import torch
import torch.nn.functional as F

from gammabayes.core.binning_geometry import GammaBinning
from gammabayes.response_functions.core.irf_kernel import make_irf_kernel
from gammabayes.utils.log_space_conv_class import LogSumExpConv3D

torch.manual_seed(1)
torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def geom():
    return GammaBinning(energy_res=6, energy_min=0.3, energy_max=30.,
                        lon_res=10, lon_min=-1.5, lon_max=1.5,
                        lat_res=10, lat_min=-1., lat_max=1.)


@pytest.fixture(scope="module")
def kernel(geom):
    return make_irf_kernel(geom, log10energy_std=torch.tensor(0.09), lon_std=torch.tensor(0.08),
                           lat_std=torch.tensor(0.08), confidence_clearance=2.)


def _cube(geom, B=3):
    """Random log-rate cube with a power-law energy slope and some -inf bins."""
    E, H, W = geom.shape
    x = (-2.5 * torch.log(geom.energy_axis))[None, :, None, None] + torch.randn(B, E, H, W) * 0.3
    x = x + torch.linspace(-3, 3, H)[None, None, :, None]       # strong spatial gradient
    mask = torch.rand(B, E, H, W) < 0.05
    x[mask] = float("-inf")
    x[:, :, 0, :] = float("-inf")                                 # a whole zero-rate row
    return x


def test_separable_matches_reference(geom, kernel):
    conv = LogSumExpConv3D(kernel_size=kernel.shape, kernel_tensor=kernel.log(), binning_geometry=geom,
                           padding=tuple(k // 2 for k in kernel.shape))
    assert conv.separable
    x = _cube(geom)
    new = conv(x)
    ref = conv.forward_reference(x)
    assert new.shape == ref.shape == x.shape
    fin = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(new), fin)
    assert torch.allclose(new[fin], ref[fin], rtol=1e-9, atol=1e-9)


def test_legacy_3d_input_and_typo_kwarg(geom, kernel):
    conv = LogSumExpConv3D(kernel_size=kernel.shape, kernel_tensor=kernel.log(), binnning_geometry=geom,
                           padding=tuple(k // 2 for k in kernel.shape))
    x = _cube(geom, B=1)
    assert conv(x[0]).shape == (1, *geom.shape)                   # old behaviour preserved


def test_chunking(geom, kernel):
    conv = LogSumExpConv3D(kernel_size=kernel.shape, kernel_tensor=kernel.log(), binning_geometry=geom,
                           padding=tuple(k // 2 for k in kernel.shape), max_batch=2)
    x = _cube(geom, B=5)
    a = conv(x); conv.max_batch = None; b = conv(x)
    fin = torch.isfinite(b)
    assert torch.equal(torch.isfinite(a), fin) and torch.allclose(a[fin], b[fin])


def test_separable_constructor_equals_log_kernel(geom, kernel):
    kE = kernel.sum((1, 2)) / kernel.sum()
    kS = kernel.sum(0) / kernel.sum(0).sum()
    a = LogSumExpConv3D(kernel_tensor=kernel.log(), binning_geometry=geom, padding=None)
    b = LogSumExpConv3D(energy_kernel=kE, spatial_kernel=kS, binning_geometry=geom, padding=None)
    x = _cube(geom)
    ra, rb = a(x), b(x)
    fin = torch.isfinite(ra)
    assert torch.allclose(ra[fin], rb[fin])


def test_per_energy_kernel(geom, kernel):
    E = geom.shape[0]
    kE = kernel.sum((1, 2)) / kernel.sum()
    kS = kernel.sum(0) / kernel.sum(0).sum()
    # all planes identical -> same as single kernel
    same = LogSumExpConv3D(energy_kernel=kE, spatial_kernel=kS[None].expand(E, -1, -1).clone(),
                           binning_geometry=geom, padding=None)
    single = LogSumExpConv3D(energy_kernel=kE, spatial_kernel=kS, binning_geometry=geom, padding=None)
    x = _cube(geom)
    r1, r2 = same(x), single(x)
    fin = torch.isfinite(r2)
    assert torch.allclose(r1[fin], r2[fin])

    # varying planes -> equals doing the spatial pass with each plane's own kernel, then the energy pass
    kh, kw = kS.shape
    widths = torch.linspace(0.5, 2.0, E)
    yy, xx = torch.meshgrid(torch.arange(kh) - kh // 2, torch.arange(kw) - kw // 2, indexing="ij")
    kS_e = torch.stack([torch.exp(-0.5 * (xx ** 2 + yy ** 2) / (w ** 2)) for w in widths])
    kS_e = kS_e / kS_e.sum((1, 2), keepdim=True)
    pe = LogSumExpConv3D(energy_kernel=kE, spatial_kernel=kS_e, binning_geometry=geom, padding=None)
    out = pe(x)
    # manual: per-plane single-kernel conv with a delta energy kernel, then compose
    delta = torch.zeros_like(kE); delta[len(kE) // 2] = 1.
    planes = []
    for e in range(E):
        c = LogSumExpConv3D(energy_kernel=delta, spatial_kernel=kS_e[e], binning_geometry=geom, padding=None)
        planes.append(c(x)[:, e])                                 # (B, H, W) spatial-only smoothed plane e
    spatial = torch.stack(planes, 1)
    # energy pass on the spatially-smoothed cube with a *delta* spatial kernel
    dS = torch.zeros(kh, kw); dS[kh // 2, kw // 2] = 1.
    ec = LogSumExpConv3D(energy_kernel=kE, spatial_kernel=dS, binning_geometry=geom, padding=None)
    manual = ec(spatial)
    fin = torch.isfinite(manual)
    assert torch.equal(torch.isfinite(out), fin)
    assert torch.allclose(out[fin], manual[fin], rtol=1e-9, atol=1e-9)


def test_energy_pass_linear_equals_logspace(geom, kernel):
    lin = LogSumExpConv3D(kernel_tensor=kernel.log(), binning_geometry=geom, padding=None)
    logp = LogSumExpConv3D(kernel_tensor=kernel.log(), binning_geometry=geom, padding=None,
                           linear_energy_max_range=0.0)       # force the unfold path
    x = _cube(geom)
    a, b = lin(x), logp(x)
    fin = torch.isfinite(b)
    assert torch.equal(torch.isfinite(a), fin) and torch.allclose(a[fin], b[fin], rtol=1e-9, atol=1e-9)
    # a cube whose energy dynamic range exceeds the threshold takes the exact path automatically
    big = x.clone(); big[:, 0] += 500.
    assert torch.allclose(lin(big)[torch.isfinite(logp(big))], logp(big)[torch.isfinite(logp(big))])


def test_gradients_flow_and_match_reference(geom, kernel):
    conv = LogSumExpConv3D(kernel_tensor=kernel.log(), binning_geometry=geom, padding=None)
    x = _cube(geom, B=2); x[~torch.isfinite(x)] = -30.      # finite everywhere for a clean gradient check
    x1 = x.clone().requires_grad_(True); x2 = x.clone().requires_grad_(True)
    conv(x1).sum().backward(); conv.forward_reference(x2).sum().backward()
    assert torch.allclose(x1.grad, x2.grad, rtol=1e-8, atol=1e-10)
