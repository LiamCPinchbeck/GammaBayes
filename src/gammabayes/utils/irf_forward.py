"""Count-conserving IRF forward operator (true space -> reconstructed space).

    obs[r] = sum_t  P(r | t) mu[t]

* ``mu`` = expected counts per true bin, on a true grid padded by the kernel
  half-width on every side (``pads``), so true events outside the analysis
  window can migrate in.
* ``P(r | t)`` = probability that an event in true bin t is reconstructed in
  bin r.  Each kernel is a bin-integrated Gaussian and sums to 1 over all r
  (``bin_prob_kernel``), so the operator conserves counts up to migration out of
  the reconstructed window (events that physically leave the field of view).
* The output is the reconstructed window only ("valid" convolution):
  (B, E_t, H_t, W_t) -> (B, E_t - 2 pE, H_t - 2 pH, W_t - 2 pW).

The PSF may depend on true energy (``spatial_kernel`` of shape (E_t, K_h, K_w)):
the spatial pass runs on the true-energy planes before energy dispersion, which
is the exact order for P(r_E, r_x | t_E, t_x) = P_E(r_E | t_E) P_x(r_x - t_x | t_E).

Inputs and outputs are log counts.  Both passes run in linear space after an
exp-shift: per (batch, true-energy plane) for the spatial pass, per (batch, pixel)
for the energy pass.  Contributions more than ~87 nats (fp32; ~708 fp64) below the
local maximum underflow to zero; this is below 1e-38 of the local rate and negligible next to the
CCR background added afterwards.  Empty windows are clamped to the dtype's smallest normal number
(finite log, zero gradient) so -inf never reaches logaddexp/autograd (0 * inf = NaN).
``linear_energy_max_range`` (default inf) forces the exact log-space unfold path
(K_d x memory) when the per-pixel energy range exceeds it; intended for testing.
"""
from __future__ import annotations
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from gammabayes.core.batching import iter_batch_chunks

_NEG_INF = float("-inf")


# ------------------------------------------------------------------ kernels
def _G(x):
    """Antiderivative of the standard normal CDF: x Phi(x) + phi(x)."""
    return x * torch.special.ndtr(x) + torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bin_prob_kernel(width, sigma, clearance=4.0):
    """P(reco bin = true bin + d), d = -n..n, for a Gaussian smear of width ``sigma``
    and a true position uniform within its bin (exact box (x) Gaussian (x) box):

        p_d = (sigma / w) [ G((d+1) w/sigma) - 2 G(d w/sigma) + G((d-1) w/sigma) ]

    ``sigma`` scalar -> (K,), or (N,) -> (N, K) on a common tap grid sized by
    max(sigma).  Truncated at ``clearance`` sigma (at least 1 bin) and renormalised
    to sum 1 per row, so truncation never loses counts."""
    # computed on the CPU in float64 (exact tails; MPS has no float64), see make_forward_irf_kernels for the cast
    sig = torch.as_tensor(sigma).detach().cpu().to(torch.float64)
    scalar = sig.ndim == 0
    sig = sig.reshape(-1).clamp(min=1e-12)
    w = float(width)
    n = max(1, int(math.ceil(clearance * float(sig.max()) / w)))
    d = torch.arange(-n, n + 1, dtype=torch.float64, device="cpu")[None, :]
    s = (w / sig)[:, None]
    p = (_G((d + 1) * s) - 2 * _G(d * s) + _G((d - 1) * s)) / s
    p = p.clamp(min=0.0)
    p = p / p.sum(1, keepdim=True)
    return p[0] if scalar else p


def make_forward_irf_kernels(log10E_width, lon_width, lat_width, log10E_std, lon_std, lat_std=None,
                             clearance=4.0, dtype=None, device=None):
    """(energy_kernel (K_d,), spatial_kernel (K_h,K_w) or (E_t,K_h,K_w)) probabilities.

    log10E_std: energy resolution in dex (sigma of log10 E_reco - log10 E_true).
    lon_std / lat_std: PSF sigma [deg]; scalars, or (E_t,) for an energy-dependent PSF.
    The kernels are returned in ``dtype`` (default: the torch default dtype) on ``device``
    (default: the torch default device).
    """
    dtype = dtype or torch.get_default_dtype()
    device = device if device is not None else torch.get_default_device()
    lat_std = lon_std if lat_std is None else lat_std
    kE = bin_prob_kernel(log10E_width, log10E_std, clearance)
    kL = bin_prob_kernel(lon_width, lon_std, clearance)
    kB = bin_prob_kernel(lat_width, lat_std, clearance)
    if kL.ndim == 1 and kB.ndim == 1:
        kS = kL[:, None] * kB[None, :]
    else:
        if kL.ndim == 1:
            kL = kL[None].expand(kB.shape[0], -1)
        if kB.ndim == 1:
            kB = kB[None].expand(kL.shape[0], -1)
        kS = kL[:, :, None] * kB[:, None, :]
    return kE.to(dtype=dtype, device=device), kS.to(dtype=dtype, device=device)


# ------------------------------------------------------------------ operator
class IRFForwardConv(nn.Module):
    """Separable, count-conserving IRF forward operator; see the module docstring for the
    conventions.  Call the instance on a ``(E_t, lon_t, lat_t)`` or ``(B, E_t, lon_t, lat_t)`` log-counts
    cube on the padded true grid to obtain log counts on the analysis grid."""


    def __init__(self, energy_kernel, spatial_kernel, max_batch=None, linear_energy_max_range=float("inf"),
                 norm_atol=1e-5):
        super().__init__()
        kE = torch.as_tensor(energy_kernel)
        kS = torch.as_tensor(spatial_kernel)
        if kE.ndim != 1 or kS.ndim not in (2, 3):
            raise ValueError("energy_kernel must be (K_d,), spatial_kernel (K_h,K_w) or (E_t,K_h,K_w)")
        if any(k % 2 == 0 for k in (kE.shape[0], kS.shape[-2], kS.shape[-1])):
            raise ValueError("kernel sizes must be odd")
        if abs(float(kE.sum()) - 1) > norm_atol or (kS.sum((-2, -1)) - 1).abs().max() > norm_atol:
            warnings.warn("IRFForwardConv: kernels do not sum to 1 -> the operator will not conserve counts")
        # obs[r] = sum_j P(-j) mu[r+j]: F.conv* is a cross-correlation, so flip the kernels
        self.register_buffer("k_energy", kE.flip(0).contiguous())
        self.register_buffer("k_spatial", kS.flip(-2, -1).contiguous())
        self.per_energy = kS.ndim == 3
        self.kernel_size = (kE.shape[0], kS.shape[-2], kS.shape[-1])
        self.pads = tuple(k // 2 for k in self.kernel_size)
        self.max_batch = max_batch
        self.linear_energy_max_range = float(linear_energy_max_range)
        self.psf_sigma_deg = None             # set by builders for diagnostics / ell_by_e

    def out_shape(self, true_shape):
        return tuple(n - 2 * p for n, p in zip(true_shape[-3:], self.pads))

    def forward(self, x):
        squeeze = x.dim() == 3
        if squeeze:
            x = x[None]
        B = x.shape[0]
        if self.max_batch is not None and B > self.max_batch:
            out = torch.cat([self._chunk(x[lo:hi]) for lo, hi in iter_batch_chunks(B, self.max_batch)], 0)
        else:
            out = self._chunk(x)
        return out[0] if squeeze else out

    def _chunk(self, x):
        B, D, H, W = x.shape
        if self.per_energy and self.k_spatial.shape[0] != D:
            raise ValueError(f"per-energy PSF has {self.k_spatial.shape[0]} planes, input has {D} true-energy bins")
        dtype, device = x.dtype, x.device
        kS = self.k_spatial.to(dtype=dtype, device=device)
        kE = self.k_energy.to(dtype=dtype, device=device)

        # spatial pass on true-energy planes ("valid": no padding)
        m = x.amax(dim=(2, 3), keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        y = torch.exp(x - m)
        if self.per_energy:
            z = F.conv2d(y, kS[:, None], groups=D)
        else:
            z = F.conv2d(y.reshape(B * D, 1, H, W), kS[None, None])
        z = z.reshape(B, D, z.shape[-2], z.shape[-1])
        tiny = torch.finfo(dtype).tiny
        a = torch.log(z.clamp_min(tiny)) + m                                   # (B, D, H_out, W_out)

        # energy pass (true -> reco, "valid"), shift per (batch, pixel)
        mE = a.amax(dim=1, keepdim=True)                                       # (B, 1, H_out, W_out)
        if self.linear_energy_max_range != float("inf"):
            rng = float((mE - a).amax().detach()) if a.numel() else 0.0
            if rng >= self.linear_energy_max_range:
                u = a.unfold(1, kE.shape[0], 1)                                # (B, D_out, H, W, K_d)
                return torch.logsumexp(u + kE.log(), dim=-1)
        zE = F.conv3d(torch.exp(a - mE)[:, None], kE[None, None, :, None, None])[:, 0]
        return torch.log(zE.clamp_min(tiny)) + mE

    def dense_reference(self, x):
        """Dense evaluation of sum_t P(r|t) mu[t] on a single small cube, for testing the fast path."""
        x = x.detach().cpu().double()                                   # CPU float64 reference
        D, H, W = x.shape
        kE = self.k_energy.flip(0).cpu().double(); kS = self.k_spatial.flip(-2, -1).cpu().double()
        pE, pH, pW = self.pads
        Do, Ho, Wo = D - 2 * pE, H - 2 * pH, W - 2 * pW
        out = torch.zeros(Do, Ho, Wo, dtype=torch.float64, device="cpu")
        mu = x.exp()
        for t in range(D):
            ks = kS[t] if self.per_energy else kS
            for de in range(-pE, pE + 1):
                r = t + de - pE
                if not 0 <= r < Do:
                    continue
                sm = F.conv2d(mu[t][None, None], ks.flip(-2, -1)[None, None])[0, 0]   # spatial smear, valid
                out[r] += kE[de + pE] * sm
        return out.log()
