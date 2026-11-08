import torch
from itertools import product


class RegularTorchInterpolator:
    """Multilinear interpolation on a regular (rectilinear) grid.

    ``points`` is a tuple of 1-d sorted axis tensors, ``values`` the gridded
    tensor with ``values.shape[i] == len(points[i])``.

    ``__call__(points_to_interp)`` takes one tensor per axis.  The query
    tensors may have *any* shape and are broadcast against each other (so a
    ``(B, 1)`` mass grid and a ``(1, E)`` energy grid evaluate on a ``(B, E)``
    mesh); the result has the broadcast shape.  Queries outside the grid are
    clamped to the edge (constant extrapolation), as before.
    """

    def __init__(self, points, values):
        self.points = tuple(points)
        self.values = values

        assert isinstance(self.values, torch.Tensor)
        self.ms = list(self.values.shape)
        self.n = len(self.points)
        assert len(self.ms) == self.n

        for i, p in enumerate(self.points):
            assert isinstance(p, torch.Tensor)
            assert p.ndim == 1 and p.shape[0] == self.values.shape[i]

    def __call__(self, points_to_interp):
        assert len(points_to_interp) == self.n

        qs = [torch.as_tensor(q) for q in points_to_interp]
        qs = torch.broadcast_tensors(*qs)
        out_shape = qs[0].shape
        qs = [q.reshape(-1).contiguous() for q in qs]

        idxs, dists, overalls = [], [], []
        for p, x in zip(self.points, qs):
            p = p.to(x.device)
            idx_right = torch.bucketize(x, p)
            idx_right = idx_right.clamp(max=p.shape[0] - 1)
            idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
            dist_left = (x - p[idx_left]).clamp(min=0.)
            dist_right = (p[idx_right] - x).clamp(min=0.)
            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left = torch.where(both_zero, torch.ones_like(dist_left), dist_left)
            dist_right = torch.where(both_zero, torch.ones_like(dist_right), dist_right)
            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        values = self.values.to(qs[0].device)
        numerator = 0.
        for indexer in product([0, 1], repeat=self.n):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator = numerator + values[tuple(as_s)] * torch.prod(torch.stack(bs_s), dim=0)
        denominator = torch.prod(torch.stack(overalls), dim=0)
        return (numerator / denominator).reshape(out_shape)
