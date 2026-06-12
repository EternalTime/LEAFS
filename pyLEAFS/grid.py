"""Shared spatial grid.

The domain is a D-dimensional box partitioned into a regular array of cubical
regions of side length ``L``, with periodic (toroidal) boundaries. Every field
and population in a simulation references one :class:`Grid`, so geometry,
neighbour lookups, and coordinate conventions are defined in exactly one place.

The grid is dimension-agnostic: ``D = 2`` or ``D = 3`` is chosen by the length
of the ``shape`` argument. Positions are always ``(n, D)`` arrays in physical
units; region indices are ``(n, D)`` integer arrays.
"""

import itertools

import numpy as np


class Grid:
    """A toroidal box partitioned into cubical regions.

    Parameters
    ----------
    shape : tuple of int
        Number of regions along each axis, e.g. ``(5, 5)`` for a 2d grid or
        ``(5, 5, 5)`` for 3d. Its length sets the dimension ``D``.
    L : float, optional
        Side length of one region in physical units (default 1.0).

    Attributes
    ----------
    shape : tuple of int
        Regions per axis.
    D : int
        Spatial dimension.
    L : float
        Region side length.
    n_regions : int
        Total number of regions, ``prod(shape)``.
    extent : ndarray of float, shape (D,)
        Physical size of the domain along each axis (``shape * L``).
    offsets : ndarray of int, shape (3**D, D)
        Moore-neighbourhood offsets, the centre cell ``(0, ..., 0)`` first.

    Examples
    --------
    >>> g = Grid((5, 5), L=10.0)
    >>> g.D, g.n_regions
    (2, 25)
    >>> g.region_of(np.array([[12.0, 3.0]]))
    array([[1, 0]])
    """

    def __init__(self, shape, L=1.0):
        self.shape = tuple(int(s) for s in shape)
        if any(s < 1 for s in self.shape):
            raise ValueError("every entry of shape must be >= 1")
        self.D = len(self.shape)
        if self.D not in (2, 3):
            raise ValueError("Grid supports D = 2 or D = 3 only")
        self.L = float(L)
        self._shape_arr = np.array(self.shape, dtype=np.int64)
        self.extent = self._shape_arr * self.L
        self.n_regions = int(np.prod(self._shape_arr))

        # Moore-neighbourhood offsets (3**D of them), centre cell first.
        deltas = list(itertools.product((0, -1, 1), repeat=self.D))
        self.offsets = np.array(deltas, dtype=np.int64)

    # ------------------------------------------------ coordinate conversions
    def wrap(self, pos):
        """Wrap physical positions into the domain by the toroidal boundary.

        Parameters
        ----------
        pos : ndarray, shape (n, D)
            Physical positions.

        Returns
        -------
        ndarray, shape (n, D)
            Positions wrapped into ``[0, extent)`` along each axis.
        """
        return np.mod(pos, self.extent)

    def region_of(self, pos):
        """Return the region index ``(n, D)`` containing each position.

        Parameters
        ----------
        pos : ndarray, shape (n, D)
            Physical positions (wrapped internally).

        Returns
        -------
        ndarray of int, shape (n, D)
            Per-axis region indices in ``[0, shape[axis])``.
        """
        idx = np.floor(self.wrap(pos) / self.L).astype(np.int64)
        # guard against floating-point landing exactly on the upper edge
        return np.mod(idx, self._shape_arr)

    def region_id(self, region_idx):
        """Flatten ``(n, D)`` region indices to linear ids ``(n,)``.

        Uses row-major (C) order consistent with ``numpy.ravel_multi_index``.
        """
        region_idx = np.asarray(region_idx)
        return np.ravel_multi_index(
            tuple(region_idx[..., d] for d in range(self.D)), self.shape
        )

    def displacement(self, a, b):
        """Minimum-image displacement ``b - a`` under periodic boundaries.

        Parameters
        ----------
        a, b : ndarray, shape (n, D)
            Physical positions.

        Returns
        -------
        ndarray, shape (n, D)
            The shortest displacement from ``a`` to ``b`` on the torus, each
            component in ``[-extent/2, extent/2)``.
        """
        d = b - a
        return d - self.extent * np.round(d / self.extent)

    def neighbour_ids(self, region_idx):
        """Linear ids of the Moore neighbourhood of each region.

        Parameters
        ----------
        region_idx : ndarray of int, shape (n, D)
            Per-axis region indices.

        Returns
        -------
        ndarray of int, shape (n, 3**D)
            Linear region ids of each region's Moore neighbourhood, the region
            itself in column 0, wrapped toroidally.
        """
        region_idx = np.asarray(region_idx)
        # (n, 1, D) + (1, 3**D, D) -> (n, 3**D, D)
        nbr = region_idx[:, None, :] + self.offsets[None, :, :]
        nbr = np.mod(nbr, self._shape_arr)
        return np.ravel_multi_index(
            tuple(nbr[..., d] for d in range(self.D)), self.shape
        )
