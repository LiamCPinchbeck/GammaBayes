"""Log-space 3-D convolution of a (B, E, lon, lat) log-rate cube with an IRF kernel.

Output bin i:

    out_i = log [ sum_j  W_j  dE_{i_d + j_d}  exp(x_{i+j}) ]  -  log [ sum_{j in domain} W_j dE_{i_d + j_d} ]

i.e. a dE-weighted (energy-integrating) convolution, renormalised per output
bin by the kernel weight that lands on in-domain input ("live weight"), so
FoV-edge bins are a well defined weighted average that doesn't depend on the
kernel clearance.

Memory: the reference implementation materialised every (K_d*K_h*K_w)-tap
patch, i.e. O(B * E * S * K^3).  ``forward`` exploits separability
W = kE (x) kLon (x) kLat and does a 2-D spatial pass (``conv2d`` in linear
space with a per-plane max shift) followed by a small log-space energy pass,
giving O(B * E * S * K_d) peak memory.  The live-weight normaliser factorises
the same way and is precomputed once per input shape.

Per-energy spatial kernels (energy-calibrated PSF) are supported: pass a
``(E, K_h, K_w)`` spatial kernel and the spatial pass becomes a grouped
``conv2d`` with one group per true-energy plane.

Numerics: both passes run in linear space after an exp-shift (per
(batch, energy) plane for the spatial pass, per batch element for the energy
pass), so values more than ~87 nats (fp32) / ~700 nats (fp64) below the
shift underflow to zero.  The energy pass measures the cube's dynamic range
and falls back to an exact log-space path above ``linear_energy_max_range``
nats, so the only approximation left is the within-plane spatial one,
beyond the usual range of rate cubes.  Like the reference, the operator is a
cross-correlation; kernels are symmetric so this is immaterial.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from gammabayes.core import GammaBinning
from gammabayes.core.batching import iter_batch_chunks

_NEG_INF = float("-inf")


def _as_tuple3(v):
    return (v, v, v) if isinstance(v, int) else tuple(v)


class LogSumExpConv3D(nn.Module):

    def __init__(self, kernel_size=None, kernel_tensor: torch.Tensor = None,
                 binning_geometry: GammaBinning = None, stride: int = 1, padding: int = 0,
                 *, energy_kernel: torch.Tensor = None, spatial_kernel: torch.Tensor = None,
                 max_batch: int | None = None, binnning_geometry: GammaBinning = None,
                 separable_rtol: float = 1e-5, verbose: bool = False,
                 linear_energy_max_range: float = 60.0):
        """
        Two ways to specify the kernel:

        1. ``kernel_tensor`` -- the LOG of a full ``(K_d, K_h, K_w)`` kernel
           (the original interface; ``make_irf_kernel(...).log()``).  It is
           factorised into separable parts; if it is not separable to
           ``separable_rtol`` the reference (unfactorised) path is used.
        2. ``energy_kernel`` ``(K_d,)`` and ``spatial_kernel`` ``(K_h, K_w)`` or
           ``(E, K_h, K_w)`` in LINEAR space (``make_irf_kernel_per_energy``).

        ``padding`` defaults to ``kernel // 2`` ("same" output) when None.
        ``max_batch`` chunks the batch axis to bound memory.
        ``linear_energy_max_range``: the energy pass runs in linear space (one
        exp-shift per batch element) while the cube's dynamic range along
        energy is below this many nats (fp32 underflows at ~87); above it the
        exact log-space unfold path (K_d x memory) is used instead.
        """
        super().__init__()
        geom = binning_geometry if binning_geometry is not None else binnning_geometry
        if geom is None:
            raise ValueError("binning_geometry is required")
        self.geom = geom
        self.verbose = verbose
        self.max_batch = max_batch
        self.linear_energy_max_range = float(linear_energy_max_range)

        # ---- kernel ------------------------------------------------------
        if kernel_tensor is not None:
            K = kernel_tensor.detach() if not isinstance(kernel_tensor, nn.Parameter) else kernel_tensor
            if kernel_size is not None and tuple(K.shape) != _as_tuple3(kernel_size):
                raise ValueError(f"Kernel tensor shape mismatch. Expected {_as_tuple3(kernel_size)}, got {tuple(K.shape)}")
            self.register_buffer("weight", K)                               # log kernel (legacy)
            lin = K.exp()
            tot = lin.sum()
            kE = lin.sum((1, 2)) / tot
            kH = lin.sum((0, 2)) / tot
            kW = lin.sum((0, 1)) / tot
            recon = tot * kE[:, None, None] * kH[None, :, None] * kW[None, None, :]
            self.separable = bool(((recon - lin).abs() <= separable_rtol * lin.max()).all())
            self.register_buffer("k_energy", (tot * kE).to(K.dtype))        # (K_d,)  linear
            self.register_buffer("k_spatial", (kH[:, None] * kW[None, :]).to(K.dtype))   # (K_h, K_w) linear
        elif energy_kernel is not None and spatial_kernel is not None:
            kE = torch.as_tensor(energy_kernel)
            kS = torch.as_tensor(spatial_kernel)
            if kS.ndim not in (2, 3):
                raise ValueError("spatial_kernel must be (K_h, K_w) or (E, K_h, K_w)")
            if kS.ndim == 3 and kS.shape[0] != len(geom.energy_axis):
                raise ValueError(f"per-energy spatial kernel has {kS.shape[0]} planes, geometry has {len(geom.energy_axis)}")
            self.register_buffer("k_energy", kE)
            self.register_buffer("k_spatial", kS)
            self.separable = True
            if kS.ndim == 2:
                full = kE[:, None, None] * kS[None]
                self.register_buffer("weight", full.log())
            else:
                self.weight = None                                           # no single full kernel exists
        else:
            raise ValueError("give kernel_tensor, or energy_kernel + spatial_kernel")

        self.per_energy = (self.k_spatial.ndim == 3)
        self.kernel_size = (int(self.k_energy.shape[0]), int(self.k_spatial.shape[-2]), int(self.k_spatial.shape[-1]))
        self.stride = _as_tuple3(stride)
        self.padding = _as_tuple3(padding) if padding is not None else tuple(k // 2 for k in self.kernel_size)
        if self.per_energy and self.weight is None and not all(s == 1 for s in self.stride):
            raise ValueError("per-energy kernels need stride 1")

        # ---- energy-bin widths on the padded axis --------------------------
        base_log10 = torch.log10(geom.energy_axis)
        diff = base_log10[1] - base_log10[0]
        p = self.padding[0]
        begin = torch.tensor([base_log10[0] - n * diff for n in range(p, 0, -1)], dtype=base_log10.dtype)
        end = torch.tensor([base_log10[-1] + n * diff for n in range(1, p + 1)], dtype=base_log10.dtype)
        self.padded_energy_axis = 10 ** torch.cat((begin, base_log10, end))
        self.log_width_correction = (self.padded_energy_axis * torch.log(torch.tensor(10.0)) * diff).log()
        self.register_buffer("log_delta_E", self.log_width_correction)

        self._live_cache = {}

    # ------------------------------------------------------------------ utils
    def _calc_output_dim(self, input_dim, kernel_dim, padding_dim, stride_dim):
        return (input_dim + 2 * padding_dim - kernel_dim) // stride_dim + 1

    def _spatial_live(self, H, W, dtype, device):
        """log of sum_{in-domain spatial taps} kLon kLat, shape (E|1, H_out, W_out)."""
        key = ("S", H, W, dtype, device)
        if key not in self._live_cache:
            kS = self.k_spatial.to(dtype=dtype, device=device)
            ones = torch.ones((1, 1, H, W), dtype=dtype, device=device)
            if self.per_energy:
                E = kS.shape[0]
                live = F.conv2d(ones.expand(1, E, H, W), kS[:, None], padding=self.padding[1:],
                                stride=self.stride[1:], groups=E)[0]                    # (E, H_out, W_out)
            else:
                live = F.conv2d(ones, kS[None, None], padding=self.padding[1:], stride=self.stride[1:])[0]  # (1, H_out, W_out)
            self._live_cache[key] = live.log()
        return self._live_cache[key]

    def _energy_live(self, D, dtype, device):
        """log of sum_{in-domain energy taps} kE dE, shape (D_out,)."""
        key = ("E", D, dtype, device)
        if key not in self._live_cache:
            p, k, s = self.padding[0], self.kernel_size[0], self.stride[0]
            valid = torch.full((D + 2 * p,), _NEG_INF, dtype=dtype, device=device)
            valid[p:p + D] = 0.
            v = (valid + self.log_delta_E.to(dtype=dtype, device=device)).unfold(0, k, s)   # (D_out, k)
            logkE = self.k_energy.to(dtype=dtype, device=device).log()
            self._live_cache[key] = torch.logsumexp(v + logkE[None], dim=1)
        return self._live_cache[key]

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.separable:
            return self.forward_reference(x)
        if x.dim() == 3:          # legacy: a 3-D cube comes back as (1, D, H, W)
            x = x[None]
        B = x.shape[0]
        if self.max_batch is not None and B > self.max_batch:
            return torch.cat([self._forward_chunk(x[lo:hi]) for lo, hi in iter_batch_chunks(B, self.max_batch)], 0)
        return self._forward_chunk(x)

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W = x.shape
        dtype, device = x.dtype, x.device
        kS = self.k_spatial.to(dtype=dtype, device=device)

        # ---- spatial pass (linear space, per-plane max shift) ------------
        m = x.amax(dim=(2, 3), keepdim=True)                                 # (B, D, 1, 1)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        y = torch.exp(x - m)                                                 # 0 where x = -inf
        if self.per_energy:
            z = F.conv2d(y, kS[:, None], padding=self.padding[1:], stride=self.stride[1:], groups=D)
        else:
            z = F.conv2d(y.reshape(B * D, 1, H, W), kS[None, None], padding=self.padding[1:], stride=self.stride[1:])
        z = z.reshape(B, D, z.shape[-2], z.shape[-1])
        log_s = torch.log(z) + m - self._spatial_live(H, W, dtype, device)[None]    # (B, D, H_out, W_out)

        # ---- energy pass -------------------------------------------------
        p, k, st = self.padding[0], self.kernel_size[0], self.stride[0]
        a = F.pad(log_s, (0, 0, 0, 0, p, p), value=_NEG_INF)                 # (B, D+2p, H_out, W_out)
        a = a + self.log_delta_E.to(dtype=dtype, device=device)[None, :, None, None]
        kE = self.k_energy.to(dtype=dtype, device=device)
        mE = a.amax(dim=(1, 2, 3), keepdim=True)                             # (B, 1, 1, 1)
        mE = torch.where(torch.isfinite(mE), mE, torch.zeros_like(mE))
        fin = torch.isfinite(a)
        lo = torch.where(fin, a, mE).amin(dim=(1, 2, 3), keepdim=True)
        rng = float((mE - lo).max().detach()) if a.numel() else 0.0
        if rng < self.linear_energy_max_range:
            # linear-space conv3d with a (k, 1, 1) kernel: O(B D H W) memory, no K_d copies
            yE = torch.exp(a - mE)
            zE = F.conv3d(yE[:, None], kE[None, None, :, None, None], stride=(st, 1, 1))[:, 0]
            num = torch.log(zE) + mE                                         # (B, D_out, H_out, W_out)
        else:
            # dynamic range along energy too large for the exp shift: exact log-space path
            u = a.unfold(1, k, st)                                           # (B, D_out, H_out, W_out, k) view
            num = torch.logsumexp(u + kE.log(), dim=-1)
        out = num - self._energy_live(D, dtype, device)[None, :, None, None]
        # fully padded column guard (-inf - -inf): cannot happen for padding = k//2, kept for generality
        return torch.where(torch.isnan(out), torch.full_like(out, _NEG_INF), out)

    # ------------------------------------------------- reference (old) path
    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Original unfactorised implementation (O(K^3) patch tensor). Kept for
        equality testing and for non-separable kernels."""
        if self.weight is None:
            raise RuntimeError("reference path needs a full log kernel (not per-energy)")
        if x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 4:
            x = x.unsqueeze(1)
        batch_size, channels, in_D, in_H, in_W = x.shape
        out_D = self._calc_output_dim(in_D, self.kernel_size[0], self.padding[0], self.stride[0])
        out_H = self._calc_output_dim(in_H, self.kernel_size[1], self.padding[1], self.stride[1])
        out_W = self._calc_output_dim(in_W, self.kernel_size[2], self.padding[2], self.stride[2])
        pad_spec = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        padded_x = F.pad(x, pad_spec, mode='constant', value=_NEG_INF).squeeze(1)
        # position-based validity volume: 0 on real input bins, -inf on padding.
        # Normalising by the live weight (not the full kernel sum) makes boundary
        # bins a clearance-stable weighted average; it MUST be position based
        # because a genuine zero-rate bin is also -inf in the data.
        valid_vol = x.new_zeros((1, 1, in_D, in_H, in_W))
        padded_valid = F.pad(valid_vol, pad_spec, mode='constant', value=_NEG_INF)[0, 0]
        conved = torch.empty((batch_size, out_D, out_H, out_W), device=x.device, dtype=x.dtype)
        k_d, k_h, k_w = self.kernel_size
        s_d, s_h, s_w = self.stride
        s_n, s_d_pad, s_h_pad, s_w_pad = padded_x.stride()
        vs_d, vs_h, vs_w = padded_valid.stride()
        weight = self.weight.to(dtype=x.dtype, device=x.device)
        ldE = self.log_width_correction.to(dtype=x.dtype, device=x.device)
        for convi in range(out_D):
            d_start = convi * s_d
            precomp = (ldE[d_start:d_start + k_d][:, None, None] + weight).flatten()[None, :, None]
            sl = padded_x[:, d_start:d_start + k_d, :, :]
            view = sl.as_strided((batch_size, k_d, out_H, k_h, out_W, k_w),
                                 (s_n, s_d_pad, s_h_pad * s_h, s_h_pad, s_w_pad * s_w, s_w_pad))
            patches = view.permute(0, 1, 3, 5, 2, 4).flatten(1, 3).flatten(2, 3)
            vsl = padded_valid[d_start:d_start + k_d, :, :]
            vview = vsl.as_strided((k_d, out_H, k_h, out_W, k_w), (vs_d, vs_h * s_h, vs_h, vs_w * s_w, vs_w))
            vpatches = vview.permute(0, 2, 4, 1, 3).flatten(0, 2).flatten(1, 2)[None]
            numerator = torch.logsumexp(patches + precomp, dim=1)
            log_live = torch.logsumexp(precomp + vpatches, dim=1)
            res = numerator - log_live
            res = torch.where(torch.isneginf(log_live), torch.full_like(res, _NEG_INF), res)
            conved[:, convi, :, :] = res.view(batch_size, out_H, out_W)
        return conved

    # ------------------------------------------------------------------ misc
    def peek(self, norm='log', vmin=None, vmax=None, *args, **kwargs):
        import numpy as np
        from matplotlib import pyplot as plt
        kE = self.k_energy.detach().cpu().numpy()
        kS = self.k_spatial.detach().cpu().numpy()
        if kS.ndim == 3:
            kS = kS.mean(0)
        fig, ax = plt.subplots(2, 2)
        ax = np.array(ax).flatten()
        ax[0].plot(kE); ax[0].set(xlabel="Energy tap", yscale=norm, ylim=[vmin, vmax])
        pcm = ax[1].pcolormesh(kS.T, norm=norm, vmin=vmin, vmax=vmax); plt.colorbar(pcm, ax=ax[1])
        ax[1].set(xlabel="Longitude tap", ylabel="Latitude tap", aspect='equal')
        ax[2].plot(kS.sum(1)); ax[2].set(xlabel="Longitude tap", yscale=norm, ylim=[vmin, vmax])
        ax[3].plot(kS.sum(0)); ax[3].set(xlabel="Latitude tap", yscale=norm, ylim=[vmin, vmax])
        plt.tight_layout()
        return fig, ax

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
