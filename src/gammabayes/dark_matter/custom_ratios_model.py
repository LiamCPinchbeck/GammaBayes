"""CustomDMRatiosModel -- multi-channel DM annihilation template.

Batched: ``eval_log_on_geom(spectral_params={"mass": m, "sigmav": sv,
"ratios": r, ["kappa": k], ["delta": d]}, spatial_params={"alpha"|"gamma": s,
["r_s": rs]})`` accepts scalars or ``(B,)`` tensors for every entry and
returns ``(B, E, lon, lat)`` (``(E, lon, lat)`` when ``B == 1`` and
``squeeze=True``).

``ratios`` may be either
  * a dict ``{channel: scalar | (B,)}`` (original interface), or
  * a tensor ``(B, n_ch)`` / ``(n_ch,)`` in ``self.channels`` order.

Spectrum-only and J-factor-only results are cached (single entry each) keyed
on the parameter values, so a call that moves only the spatial parameters
does not re-evaluate the spectra and vice versa.
"""
import torch
from functools import partial, reduce
from gammabayes.priors import TwoCompFluxPrior
from gammabayes.dark_matter import PPPCReader, SingleDMSpectralComp
from gammabayes.dark_matter.density_profiles import DM_Profile, Einasto_Profile
from gammabayes.core.batching import as_param, batch_size


def _needs_grad(*vals):
    return any(torch.is_tensor(v) and v.requires_grad for v in vals)


def _param_key(**kw):
    parts = []
    for k in sorted(kw):
        v = kw[k]
        if v is None:
            parts.append((k, None))
        elif torch.is_tensor(v):
            parts.append((k, tuple(v.shape), v.detach().cpu().double().numpy().tobytes()))
        else:
            parts.append((k, float(v)))
    return tuple(parts)


class CustomDMRatiosModel(TwoCompFluxPrior):
    """Custom DM channel-ratio model; batched spectral + spatial evaluation."""

    def __init__(self,
                 binning_geometry=None,
                 mass=torch.tensor(1.),
                 sigmav=3e-26,
                 spatial_profile=Einasto_Profile,
                 spatial_profile_kwargs=None,
                 channels=['WW', 'ZZ', 'HH', 'tt'],
                 ratios=None,
                 symmetry_factor=torch.tensor(1.),
                 spectral_reader='cosmixs',
                 cache: bool = True,
                 cache_max_batch: int = 16,
                 *args, **kwargs):

        all_channels = list(PPPCReader.darkSUSY_to_PPPC_converter.keys())
        self.is_single_channel = False

        if isinstance(channels, (list, tuple)):
            incorrect_channels = [ch for ch in channels if ch not in all_channels]
            if len(incorrect_channels):
                raise ValueError(f"Invalid channels given ->{incorrect_channels} must be one or "
                                 f"multiple of the following channels: {all_channels}")
            self.channels = list(channels)
        else:
            raise ValueError("Currently require that the channels input is a list of strings for the channels")

        self.spectral_models = {}
        for channel in self.channels:
            self.spectral_models[channel] = SingleDMSpectralComp(
                binning_geometry=binning_geometry, channel=channel, mass=mass,
                reader=spectral_reader)

        self.constant_spectral_prefactor = torch.tensor(1 / (8 * torch.pi))
        self.use_cache = cache
        # Only cache small batches: a batched caller varies every parameter per
        # call, so a hit is impossible and the (B, S) J cube would just sit on
        # the GPU.  The B == 1 loop pattern is where the cache pays off.
        self.cache_max_batch = int(cache_max_batch)
        self._spec_cache = None
        self._spat_cache = None

        self.spectral_comp_logfunc = partial(self._spectral_comp_logfunc,
                                             mass=mass, sigmav=sigmav, ratios=ratios,
                                             symmetry_factor=symmetry_factor)
        if isinstance(spatial_profile, type):          # a profile CLASS -> instantiate (defaults if no kwargs)
            spatial_profile = spatial_profile(**(spatial_profile_kwargs or {}))
        elif spatial_profile_kwargs:
            raise ValueError("spatial_profile_kwargs given but spatial_profile is already an instance")
        # TwoCompFluxPrior overwrites self.spatial_comp with the cached wrapper,
        # so the profile itself lives under spatial_profile.
        self.spatial_profile = spatial_profile

        super().__init__(
            binning_geometry=binning_geometry,
            spectral_comp=self._cached_spectral,
            spatial_comp=self._cached_spatial,
            **kwargs)

    # ------------------------------------------------------------------ ratios
    def _ratios_tensor(self, ratios, like):
        """-> (B, n_ch) tensor in self.channels order."""
        if ratios is None:
            n = len(self.channels)
            return torch.full((1, n), 1.0 / n, dtype=like.dtype, device=like.device)
        if torch.is_tensor(ratios):
            r = ratios.to(like.device)
            if r.ndim == 1:
                if r.shape[0] != len(self.channels):
                    raise ValueError(f"ratios has {r.shape[0]} entries, expected {len(self.channels)}")
                r = r[None]
            elif r.ndim != 2 or r.shape[1] != len(self.channels):
                raise ValueError(f"ratios must be (B, {len(self.channels)}), got {tuple(r.shape)}")
            return r
        if isinstance(ratios, dict):
            cols = [as_param(ratios[ch], 0, like=like) for ch in self.channels]
            dt = reduce(torch.promote_types, [c.dtype for c in cols])
            B = batch_size(*cols)
            return torch.stack([c.to(dt).expand(B) for c in cols], dim=1)
        raise TypeError("ratios must be a dict, a tensor, or None")

    # ---------------------------------------------------------------- spectral
    def _multi_channel_spectral_comp(self, energy, mass, ratios, kappa=None, delta=None):
        """energy (E,), mass (B,), ratios (B, n_ch) -> (B, E)."""
        log_specs = torch.stack([self.spectral_models[ch](energy, mass=mass, kappa=kappa, delta=delta)
                                 for ch in self.channels], dim=0)                       # (n_ch, B, E)
        log_ratios = torch.log(ratios.clamp(min=1e-37)).T[:, :, None]                  # (n_ch, B, 1); 1e-37 is fp32-safe
        return torch.logsumexp(log_ratios + log_specs, dim=0)

    def _spectral_comp_logfunc(self, energy, mass, ratios, sigmav, symmetry_factor, kappa=None, delta=None):
        E = torch.as_tensor(energy)
        mass_b = as_param(mass, 1, like=E)                                              # (B, 1)
        sigmav_b = as_param(sigmav, 1, like=E)
        sym_b = as_param(symmetry_factor, 1, like=E)
        ratios_t = self._ratios_tensor(ratios, like=E)
        logprefactor = torch.log(sigmav_b * self.constant_spectral_prefactor / (sym_b * mass_b ** 2))
        return logprefactor + self._multi_channel_spectral_comp(E, mass=mass_b[:, 0], ratios=ratios_t,
                                                                kappa=kappa, delta=delta)

    def _cached_spectral(self, energy, **spectral_params):
        # The cache is bypassed whenever an input carries autograd state: a cached
        # output belongs to an earlier graph and would route gradients to the wrong
        # (or freed) tensors.
        flat = {}
        for k, v in spectral_params.items():
            if isinstance(v, dict):                       # ratios as a dict
                flat.update({f"{k}.{kk}": vv for kk, vv in v.items()})
            else:
                flat[k] = v
        # The energy axis is part of the cache key but not of the batch-size check
        # (it is (E,), not a batch parameter; including it raised 'inconsistent batch sizes').
        if (not self.use_cache or _needs_grad(energy, *flat.values())
                or batch_size(*flat.values()) > self.cache_max_batch):
            return self.spectral_comp_logfunc(energy, **spectral_params)
        key = _param_key(__energy__=energy, **flat)
        if self._spec_cache is not None and self._spec_cache[0] == key:
            return self._spec_cache[1]
        out = self.spectral_comp_logfunc(energy, **spectral_params)
        self._spec_cache = (key, out)
        return out

    # ----------------------------------------------------------------- spatial
    def _cached_spatial(self, lonlatgrid, **spatial_params):
        if not self.use_cache or _needs_grad(*spatial_params.values()) \
                or batch_size(*spatial_params.values()) > self.cache_max_batch:
            return self.spatial_profile(lonlatgrid, **spatial_params)
        key = _param_key(__grid_shape__=torch.tensor(lonlatgrid.shape[:-1]), **spatial_params)
        if self._spat_cache is not None and self._spat_cache[0] == key:
            return self._spat_cache[1]
        out = self.spatial_profile(lonlatgrid, **spatial_params)
        self._spat_cache = (key, out)
        return out

    def clear_cache(self):
        self._spec_cache = None
        self._spat_cache = None

    # ---------------------------------------------------------------- helpers
    def log_spectrum(self, energy=None, **spectral_params):
        """(B, E) log dN/dE x prefactor, on the geometry's energy axis by default."""
        if energy is None:
            energy = self.binning_geometry.energy_axis
        return self._cached_spectral(energy, **spectral_params)

    def log_J(self, **spatial_params):
        """(B, lon, lat) log differential J-factor on the geometry's sky grid."""
        return self._cached_spatial(self.binning_geometry.shaped_grid[0, ..., 1:], **spatial_params)

    def __iter__(self):
        return iter(self.channels)

    def keys(self):
        return list(self.channels)
