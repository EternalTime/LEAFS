"""Generic spatial hash over a point set.

A :class:`SpatialHash` buckets an arbitrary ``(n, D)`` set of points into the
regions of a shared :class:`Grid`, then answers Moore-neighbourhood queries.
It is deliberately agnostic about what the points are --- resources, foragers,
predators --- so the same structure serves every layer. Unlike the resource
field's internal buffer, this is a lightweight index rebuilt from a positions
array on demand.
"""

import numpy as np

from pyLEAFS.grid import Grid


class SpatialHash:
    """Bucket a point set into grid regions for neighbourhood queries.

    Parameters
    ----------
    grid : Grid
        Shared spatial grid.
    positions : ndarray, shape (n, D), optional
        Initial point set. May be rebuilt later with :meth:`build`.

    Examples
    --------
    >>> g = Grid((5, 5), L=10.0)
    >>> pts = np.array([[1.0, 1.0], [12.0, 3.0], [1.5, 1.5]])
    >>> sh = SpatialHash(g, pts)
    >>> idx = sh.query(np.array([1.2, 1.2]))
    >>> sorted(idx.tolist())
    [0, 2]
    """

    def __init__(self, grid, positions=None):
        self.grid = grid
        if positions is not None:
            self.build(positions)
        else:
            self._positions = np.zeros((0, grid.D))
            self._buckets = {}

    def build(self, positions):
        """(Re)index ``positions`` into per-region buckets.

        Parameters
        ----------
        positions : ndarray, shape (n, D)
            Physical positions to index.
        """
        self._positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
        if self._positions.shape[0] == 0:
            self._buckets = {}
            return
        rids = self.grid.region_id(self.grid.region_of(self._positions))
        self._buckets = {}
        for i, rid in enumerate(rids):
            self._buckets.setdefault(int(rid), []).append(i)

    def query(self, pos):
        """Indices of points in the Moore neighbourhood of ``pos``.

        Parameters
        ----------
        pos : ndarray, shape (D,)
            Physical query position.

        Returns
        -------
        ndarray of int
            Indices into the indexed positions array of every point whose
            region is in the Moore neighbourhood of ``pos``. Not distance
            filtered --- the caller refines with actual displacements.
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(1, self.grid.D)
        ridx = self.grid.region_of(pos)
        nbr_ids = self.grid.neighbour_ids(ridx)[0]
        out = []
        for rid in nbr_ids:
            b = self._buckets.get(int(rid))
            if b:
                out.extend(b)
        return np.array(out, dtype=np.int64)

    def query_within(self, pos, radius):
        """Indices of indexed points within ``radius`` of ``pos``.

        Refines :meth:`query` by minimum-image distance. Assumes ``radius`` is
        no larger than one region side so the Moore neighbourhood suffices.
        """
        cand = self.query(pos)
        if cand.size == 0:
            return cand
        disp = self.grid.displacement(
            np.asarray(pos, dtype=np.float64).reshape(1, self.grid.D),
            self._positions[cand],
        )
        d2 = np.einsum("ij,ij->i", disp, disp)
        return cand[d2 <= radius * radius]
