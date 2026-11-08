import math
import torch
from gammabayes.utils import IRFForwardConv, bin_prob_kernel, make_forward_irf_kernels

torch.set_default_dtype(torch.float64)


def test_kernel_sums_to_one_and_symmetric():
    for w, s in [(0.1, 0.09), (0.1, 0.02), (0.1, 0.3)]:
        p = bin_prob_kernel(w, s, 4.0)
        assert abs(float(p.sum()) - 1) < 1e-12
        assert torch.allclose(p, p.flip(0))
    P = bin_prob_kernel(0.1, torch.tensor([0.02, 0.08, 0.2]), 4.0)
    assert P.shape[0] == 3 and torch.allclose(P.sum(1), torch.ones(3))


def _cube(D, H, W, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(D, H, W, generator=g) * 2 - 1


def test_matches_dense_reference_scalar_psf():
    kE, kS = make_forward_irf_kernels(0.1, 0.1, 0.1, 0.09, 0.08, clearance=3.0)
    op = IRFForwardConv(kE, kS)
    x = _cube(14, 12, 11)
    assert torch.allclose(op(x), op.dense_reference(x), atol=1e-10)


def test_matches_dense_reference_per_energy_psf():
    D = 14
    sig = torch.linspace(0.03, 0.2, D)
    kE, kS = make_forward_irf_kernels(0.1, 0.1, 0.1, 0.09, sig, clearance=3.0)
    op = IRFForwardConv(kE, kS)
    x = _cube(D, 15, 15, seed=1)
    assert torch.allclose(op(x), op.dense_reference(x), atol=1e-10)


def test_count_conservation_interior():
    """A source well inside the window keeps all its counts."""
    kE, kS = make_forward_irf_kernels(0.1, 0.1, 0.1, 0.09, 0.08, clearance=4.0)
    op = IRFForwardConv(kE, kS)
    pE, pH, pW = op.pads
    D, H, W = 30 + 2 * pE, 30 + 2 * pH, 30 + 2 * pW
    mu = torch.zeros(D, H, W)
    mu[D // 2, H // 2, W // 2] = 1e4
    out = op(mu.clamp(min=1e-300).log()).exp()
    assert abs(float(out.sum()) / 1e4 - 1) < 1e-9


def test_batched_equals_loop_and_grad():
    kE, kS = make_forward_irf_kernels(0.1, 0.1, 0.1, 0.09, 0.08, clearance=3.0)
    op = IRFForwardConv(kE, kS, max_batch=2)
    x = torch.stack([_cube(12, 10, 10, seed=s) for s in range(5)]).requires_grad_(True)
    y = op(x)
    for b in range(5):
        assert torch.allclose(y[b], op(x[b].detach()), atol=1e-12)
    y.exp().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_log_space_fallback_matches_linear():
    kE, kS = make_forward_irf_kernels(0.1, 0.1, 0.1, 0.09, 0.08, clearance=3.0)
    x = _cube(12, 10, 10, seed=3)
    x[:3] -= 80.0                                    # > 60 nats energy dynamic range
    a = IRFForwardConv(kE, kS, linear_energy_max_range=1e9)(x)
    b = IRFForwardConv(kE, kS, linear_energy_max_range=1.0)(x)
    assert torch.allclose(a, b, atol=1e-9)


def test_binning_from_edges_matches_constructor():
    from gammabayes import GammaBinning
    g = GammaBinning(energy_res=10, energy_min=0.1, energy_max=100., lon_res=10, lon_min=-3., lon_max=3.,
                     lat_res=10, lat_min=-3., lat_max=3.)
    h = GammaBinning.from_edges(g.energy_edges, g.lon_edges, g.lat_edges)
    for a, b in zip(g.axes, h.axes):
        assert torch.allclose(a, b)
    assert torch.allclose(g.bin_width_mat, h.bin_width_mat)
    assert h.shaped_grid.shape == g.shaped_grid.shape


def test_empty_windows_finite_values_and_grads():
    """-inf inputs (e.g. DM above its mass) must not give NaN gradients through logaddexp."""
    kE, kS = make_forward_irf_kernels(0.1, 0.1, 0.1, 0.09, 0.08, clearance=3.0)
    op = IRFForwardConv(kE, kS)
    x = _cube(14, 12, 12, seed=4)
    x[7:] = float("-inf")
    x = x[None].repeat(2, 1, 1, 1).requires_grad_(True)
    y = torch.logaddexp(op(x), torch.zeros(()))
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
