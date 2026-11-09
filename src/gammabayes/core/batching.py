"""Batch-parameter helpers.

How batching works throughout the priors / dark_matter components:

* A *parameter* may be a python scalar, a 0-d tensor, or a 1-d tensor of
  shape ``(B,)``.  Several parameters passed together are broadcast to a
  common ``B`` (scalars act as ``B = 1``).
* A component's log-function returns a tensor with a leading batch axis:
  spectral ``(B, E)``, spatial ``(B, lon, lat)``, full rate ``(B, E, lon, lat)``.
* Static (parameter-free) templates return the *unbatched* ``(E, lon, lat)``
  (or ``(lon, lat)``) tensor; ``TwoCompFluxPrior`` promotes them with a
  leading ``1``.
* ``DiscreteLogPrior.eval_log_on_geom(squeeze=True)`` drops the batch axis
  when ``B == 1`` so pre-batching call sites keep working unchanged.
"""
import torch


def as_param(x, n_trailing: int, *, like: torch.Tensor | None = None) -> torch.Tensor:
    """Return ``x`` as a tensor of shape ``(B, 1, ..., 1)`` with ``n_trailing``
    singleton axes, ready to broadcast against an axis-shaped tensor.

    ``B`` is 1 for scalars, ``x.shape[0]`` for 1-d input.  Inputs with more
    than one dimension are an error (batch parameters are 0-d or 1-d).
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=like.dtype if like is not None else None,
                            device=like.device if like is not None else None)
    elif like is not None:
        x = x.to(device=like.device)
    if x.ndim == 0:
        x = x.reshape(1)
    elif x.ndim > 1:
        raise ValueError(f"batch parameters must be 0-d or 1-d, got shape {tuple(x.shape)}")
    return x.reshape(x.shape[0], *([1] * n_trailing))


def batch_size(*params) -> int:
    """Common batch size of a set of parameters (scalars count as 1)."""
    B = 1
    for p in params:
        if p is None:
            continue
        if torch.is_tensor(p):
            n = p.shape[0] if p.ndim >= 1 else 1
        elif hasattr(p, "__len__") and not isinstance(p, (str, bytes)):
            n = len(p)
        else:
            n = 1
        if n != 1:
            if B != 1 and n != B:
                raise ValueError(f"inconsistent batch sizes {B} and {n}")
            B = n
    return B


def promote_batch(x: torch.Tensor, n_axes: int) -> torch.Tensor:
    """Give ``x`` a leading batch axis if it has exactly ``n_axes`` axes
    (i.e. it is an unbatched axis-shaped tensor)."""
    return x.unsqueeze(0) if x.ndim == n_axes else x


def iter_batch_chunks(B: int, max_batch: int | None):
    """Yield ``(lo, hi)`` slices covering ``range(B)`` in chunks."""
    if max_batch is None or max_batch <= 0 or max_batch >= B:
        yield 0, B
        return
    for lo in range(0, B, max_batch):
        yield lo, min(B, lo + max_batch)
