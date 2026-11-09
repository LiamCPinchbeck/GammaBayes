from gammabayes.core.batching import as_param, promote_batch


class BaseSpectral_PriorComp:
    """Wraps a log-spectral function ``logfunc(energy, **params)``.

    Batching: ``energy`` is the 1-d axis ``(E,)``; each parameter is a
    scalar or ``(B,)`` tensor; the return value is ``(B, E)`` (``B = 1`` for
    all-scalar parameters).  Subclasses should build their functions with
    ``as_param(p, 1)`` so that parameters broadcast as ``(B, 1)`` against
    ``energy[None, :]``.
    """

    def __init__(self, logfunc, *args, **kwargs):
        self._input_logfunc = logfunc

    def __call__(self, energy, *args, **kwargs):
        out = self._input_logfunc(energy, *args, **kwargs)
        return promote_batch(out, 1)
