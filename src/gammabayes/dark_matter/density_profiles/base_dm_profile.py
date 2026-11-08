import torch
from torch.utils.checkpoint import checkpoint
from gammabayes import haversine
from gammabayes.core.batching import as_param, batch_size, iter_batch_chunks
from gammabayes.priors.spatial_components import BaseSpatial_PriorComp
from functools import partial


class DM_Profile(BaseSpatial_PriorComp):
    """Dark-matter density profile and line-of-sight (J-factor) integral.

    Batched: every profile keyword (``alpha``, ``gamma``, ``r_s`` ...) may be
    a scalar or a ``(B,)`` tensor; ``logdiffJ`` returns ``(B, lon, lat)``.
    The LOS integral is ``(B, S, N_los)`` in memory, so it is evaluated in
    chunks of at most ``max_elems`` elements (default 2e8 ~ 0.8 GB fp32).
    """
    d_kpc_to_cm = torch.tensor(3.086e21)
    sr_to_deg2 = torch.tensor(3.0461741978670857e-4)
    theta_min_deg = 1e-3          # floor for the impact parameter (b = 0 at a pixel centre)

    def __init__(self,
                 log_profile_func: callable,
                 rho_s=torch.tensor(1.),
                 LOCAL_DENSITY=torch.tensor(3.9e-4),      # TeV/cm3
                 dist_to_source=torch.tensor(8.3),        # kpc
                 annihilation=torch.tensor(1.),
                 angular_central_coords=torch.tensor([0, 0]),   # deg
                 int_resolution: int = 101,
                 max_elems: int = int(2e8),
                 grad_checkpoint: bool = True,
                 *args, **kwargs):

        self._log_profile_func = log_profile_func
        self.int_resolution = int(int_resolution)
        self.max_elems = int(max_elems)
        # With autograd live, every chunk's (Bc, S, N) intermediates would be kept
        # for backward, i.e. chunking would not bound memory at all.  Checkpointing
        # recomputes the chunk forward during backward instead (2x LOS cost).
        self.grad_checkpoint = bool(grad_checkpoint)
        self.t_range = torch.linspace(0, 4, self.int_resolution)
        self.deltat = self.t_range[1] - self.t_range[0]

        _dev = self.t_range.device
        self.LOCAL_DENSITY = torch.as_tensor(LOCAL_DENSITY).to(_dev)
        self.DISTANCE = torch.as_tensor(dist_to_source).to(_dev)
        self.annihilation = torch.as_tensor(annihilation).to(_dev)
        self.rho_s = torch.as_tensor(rho_s).to(_dev)
        self.angular_central_coords = torch.as_tensor(angular_central_coords).to(_dev)
        self.d_kpc_to_cm = self.d_kpc_to_cm.to(_dev)
        self.sr_to_deg2 = self.sr_to_deg2.to(_dev)

        self.log_integral_constants = torch.log(self.deltat * self.d_kpc_to_cm * self.sr_to_deg2).to(self.DISTANCE.device)

        self.log_profile_func = self.scale_density_profile_to_refs(
            ref_density=self.LOCAL_DENSITY, ref_density_radius=self.DISTANCE,
            rho_s=rho_s, log_profile_func=self._log_profile_func, *args, **kwargs)

        super().__init__(logfunc=self.logdiffJ)

    @staticmethod
    def calc_new_scale_density_to_refs(ref_density, ref_density_radius, rho_s, log_profile_func, *args, **kwargs):
        log_density = log_profile_func(ref_density_radius, rho_s=rho_s, *args, **kwargs)
        density_ref_ratio = ref_density / log_density.exp()
        return rho_s * density_ref_ratio

    @staticmethod
    def scale_density_profile_to_refs(ref_density, ref_density_radius, rho_s, log_profile_func, *args, **kwargs):
        rho_s = DM_Profile.calc_new_scale_density_to_refs(
            ref_density=ref_density, ref_density_radius=ref_density_radius, log_profile_func=log_profile_func,
            rho_s=rho_s, *args, **kwargs)
        return partial(log_profile_func, rho_s=rho_s, *args, **kwargs)

    # ------------------------------------------------------------------------
    def logdiffJ(self, lonlatgrid, **kwargs):
        lon_grid = lonlatgrid[..., 0]
        lat_grid = lonlatgrid[..., 1]
        angular_offset = haversine(lon_grid.flatten(), lat_grid.flatten(),
                                   self.angular_central_coords[0], self.angular_central_coords[1])
        out = self._log_diffJ_from_offset(theta=angular_offset, kwargs=kwargs)      # (B, S)
        return out.reshape(out.shape[0], *lat_grid.shape)

    def _log_diffJ_from_offset(self, theta, kwargs):
        """theta (S,) in deg; kwargs scalar or (B,) -> (B, S) log dJ/dOmega.

        LOS parametrised by angle from closest approach: s = s0 + b tan u,
        ds = b sec^2(u) du, r = b sec(u).  Smooth integrand -> N midpoints are
        accurate to <0.5% at all offsets.
        """
        th = torch.deg2rad(theta)
        th_min = torch.deg2rad(torch.as_tensor(self.theta_min_deg, dtype=th.dtype, device=th.device))
        th = th.clamp(min=th_min)                       # b = 0 only at a pixel exactly on the GC
        D = self.DISTANCE.to(th.device)
        b = D * torch.sin(th)                           # impact parameter (kpc)
        s0 = D * torch.cos(th)                          # distance along LOS to closest approach
        s_max = float(self.t_range[-1]) * D             # outer boundary 4D
        u_lo = torch.atan(-s0 / b)                      # observer, s = 0
        u_hi = torch.atan((s_max - s0) / b)
        N = self.int_resolution
        frac = (torch.arange(N, device=th.device, dtype=th.dtype) + 0.5) / N
        u = u_lo[:, None] + (u_hi - u_lo)[:, None] * frac[None, :]           # (S, N)
        log_du = torch.log((u_hi - u_lo) / N)[:, None]
        cos_u = torch.cos(u)
        r = b[:, None] / cos_u                                                # (S, N)
        log_ds = torch.log(b)[:, None] - 2. * torch.log(cos_u) + log_du      # ds = b sec^2 u du
        log_const = torch.log(self.d_kpc_to_cm * self.sr_to_deg2).to(th.device)

        # batch the profile kwargs as (B, 1, 1) so they broadcast against r (1, S, N)
        kw_b = {k: as_param(v, 2, like=th) for k, v in kwargs.items()}
        B = batch_size(*[v[:, 0, 0] for v in kw_b.values()]) if kw_b else 1
        S = th.shape[0]

        # re-anchor rho(r_sun) = LOCAL_DENSITY for the CURRENT shape kwargs
        log_norm_corr = (torch.log(self.LOCAL_DENSITY.to(th.device))
                         - self.log_profile_func(D, **{k: v[:, :, 0] for k, v in kw_b.items()}))   # (B, 1)
        log_norm_corr = log_norm_corr.reshape(-1, 1, 1).expand(B, 1, 1)

        max_batch = max(1, self.max_elems // max(1, S * N))
        ann = float(self.annihilation)
        keys = list(kw_b)

        def _chunk(lnc, *vals):                       # positional so checkpoint can track grads
            kw_c = dict(zip(keys, vals))
            log_rho_pow = (1. + ann) * (self.log_profile_func(r[None], **kw_c) + lnc)
            return torch.logsumexp(log_rho_pow + log_ds[None], dim=-1) + log_const

        use_ckpt = (self.grad_checkpoint and torch.is_grad_enabled()
                    and any(v.requires_grad for v in kw_b.values()))
        chunks = []
        for lo, hi in iter_batch_chunks(B, max_batch):
            vals = [(v[lo:hi] if v.shape[0] > 1 else v) for v in kw_b.values()]
            lnc = log_norm_corr[lo:hi]
            if use_ckpt:
                chunks.append(checkpoint(_chunk, lnc, *vals, use_reentrant=False))
            else:
                chunks.append(_chunk(lnc, *vals))
        return torch.cat(chunks, 0) if len(chunks) > 1 else chunks[0]
