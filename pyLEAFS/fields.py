"""Fields living on the shared grid.

A *field* is anything spatially distributed that the simulation steps in time
and that agents can sample: the resource field here, and later a pheromone
field, chemical gradients, and so on. The simulation holds a *list* of fields
and steps each one, so new field types are add-ons rather than rewrites.

:class:`Field` is the structural protocol every field follows. :class:`ResourceField`
is the v1 implementation: a replenishing Poisson point process, ported from the
MATLAB ``Region``/``Environment`` classes but stored as one flat struct-of-arrays
buffer instead of per-region objects.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from pyLEAFS.grid import Grid


@runtime_checkable
class Field(Protocol):
    """Structural protocol for a steppable, samplable field.

    A field references a :class:`Grid` and implements the three verbs the
    simulation and agents use. ``deposit`` is optional in spirit (the resource
    field does not accept deposits), but the signature is reserved so that
    pheromone-style fields slot in without changing call sites.
    """

    grid: Grid

    def step(self, rng) -> None:
        """Advance the field one timestep using random source ``rng``."""
        ...

    def sample(self, pos):
        """Return whatever an agent senses of the field at positions ``pos``."""
        ...

    def deposit(self, pos, amount) -> None:
        """Add ``amount`` to the field at positions ``pos`` (if supported)."""
        ...


class ResourceField:
    """A replenishing Poisson resource field on the shared grid.

    Resources are points scattered through the domain. Each timestep, within
    every region, new resources are born and existing ones die:

    .. math::

        k_\\mathrm{grow}  &\\sim \\mathrm{Poisson}(\\Gamma L^D\\,dt / \\epsilon) \\\\
        k_\\mathrm{decay} &\\sim \\mathrm{Binomial}(N,\\, 1 - e^{-\\gamma\\,dt})

    giving an agent-free equilibrium of :math:`N_\\mathrm{eq} = \\Gamma L^D /
    (\\epsilon\\gamma)` resources per region. Positions are stored in one flat
    ``(capacity, D)`` buffer; each region owns a contiguous slot range and an
    active count, so births append, deaths swap-delete, and harvest is O(1) ---
    the struct-of-arrays equivalent of the MATLAB active-prefix ``Region``.

    Parameters
    ----------
    grid : Grid
        Shared spatial grid.
    Gamma : float
        Resource birth rate per unit volume, :math:`\\Gamma`.
    gamma : float
        Resource decay rate, :math:`\\gamma`.
    epsilon : float
        Energy yield per resource, :math:`\\epsilon`.
    dt : float
        Timestep.
    capacity_factor : float, optional
        Per-region buffer is ``max(20, ceil(capacity_factor * N_eq))`` slots
        (default 4.0, matching the MATLAB ``Region``).

    Attributes
    ----------
    N_eq : float
        Agent-free equilibrium count per region.
    count : ndarray of int, shape (n_regions,)
        Current number of active resources in each region.
    positions : ndarray, shape (n_active, D)
        Physical positions of all active resources (rebuilt on demand by
        :meth:`all_positions`).

    Examples
    --------
    >>> g = Grid((5, 5), L=10.0)
    >>> rf = ResourceField(g, Gamma=0.001, gamma=0.1, epsilon=0.1, dt=0.025)
    >>> round(rf.N_eq)
    10
    """

    def __init__(self, grid, Gamma, gamma, epsilon, dt, capacity_factor=4.0):
        self.grid = grid
        self.Gamma = float(Gamma)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.dt = float(dt)

        L, D = grid.L, grid.D
        self.N_eq = self.Gamma * L**D / (self.epsilon * self.gamma)
        self._lam = self.Gamma * L**D * self.dt / self.epsilon       # Poisson rate
        self._p = 1.0 - np.exp(-self.gamma * self.dt)                # death prob

        nreg = grid.n_regions
        self._cap = max(20, int(np.ceil(capacity_factor * self.N_eq)))
        # Flat struct-of-arrays buffer: region r owns slots [r*cap, r*cap+cap).
        # _local holds normalised [0, 1)^D positions within the region.
        self._local = np.zeros((nreg * self._cap, D))
        self.count = np.zeros(nreg, dtype=np.int64)

        # Precompute each region's lower-left physical corner, (n_regions, D).
        region_idx = np.array(
            np.unravel_index(np.arange(nreg), grid.shape)
        ).T.astype(np.float64)
        self._corner = region_idx * L

    # ----------------------------------------------------------- population
    def seed(self, rng, n0=None):
        """Populate every region near equilibrium.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random source.
        n0 : int, optional
            Exact initial count per region. If omitted, each region draws
            ``Poisson(N_eq)``.
        """
        nreg = self.grid.n_regions
        if n0 is None:
            counts = rng.poisson(self.N_eq, size=nreg)
        else:
            counts = np.full(nreg, int(n0))
        counts = np.minimum(counts, self._cap)
        self.count[:] = 0
        for r in range(nreg):
            self._grow_region(r, int(counts[r]), rng)

    def _grow_region(self, r, k, rng):
        """Append ``k`` uniformly-placed resources to region ``r``."""
        if k <= 0:
            return
        n = self.count[r]
        k = min(k, self._cap - n)
        if k <= 0:
            return
        base = r * self._cap
        self._local[base + n : base + n + k] = rng.random((k, self.grid.D))
        self.count[r] = n + k

    def _decay_region(self, r, k, rng):
        """Remove ``k`` uniformly-chosen resources from region ``r``."""
        n = self.count[r]
        if n == 0 or k <= 0:
            return
        k = min(k, n)
        base = r * self._cap
        # choose k survivors to keep compact: drop k random indices
        drop = rng.choice(n, size=k, replace=False)
        keep = np.ones(n, dtype=bool)
        keep[drop] = False
        self._local[base : base + n - k] = self._local[base : base + n][keep]
        self.count[r] = n - k

    # --------------------------------------------------------------- step
    def step(self, rng):
        """Advance resource dynamics one timestep.

        Birth and death counts for all regions are drawn in two vectorized RNG
        calls; grow/decay order is randomised per region to avoid bias.
        """
        nreg = self.grid.n_regions
        k_grow = rng.poisson(self._lam, size=nreg)
        k_decay = rng.binomial(self.count, self._p)
        grow_first = rng.random(nreg) > 0.5
        for r in range(nreg):
            if grow_first[r]:
                self._grow_region(r, int(k_grow[r]), rng)
                self._decay_region(r, int(k_decay[r]), rng)
            else:
                self._decay_region(r, int(k_decay[r]), rng)
                self._grow_region(r, int(k_grow[r]), rng)

    # ------------------------------------------------------------ queries
    def all_positions(self):
        """Return physical positions of every active resource, ``(M, D)``."""
        cap, D = self._cap, self.grid.D
        chunks = []
        for r in range(self.grid.n_regions):
            n = self.count[r]
            if n:
                base = r * cap
                chunks.append(self._corner[r] + self._local[base : base + n] * self.grid.L)
        if not chunks:
            return np.zeros((0, D))
        return np.concatenate(chunks, axis=0)

    @property
    def positions(self):
        return self.all_positions()

    def total(self):
        """Total active resource count across the domain."""
        return int(self.count.sum())

    def _region_slice(self, r):
        base = r * self._cap
        n = self.count[r]
        return base, n

    def neighbourhood(self, pos):
        """Resources in the Moore neighbourhood of a single query position.

        Parameters
        ----------
        pos : ndarray, shape (D,)
            Physical query position.

        Returns
        -------
        disp : ndarray, shape (m, D)
            Minimum-image displacement from ``pos`` to each nearby resource.
        region_ids : ndarray of int, shape (m,)
            Owning region id for each resource (for :meth:`harvest`).
        slots : ndarray of int, shape (m,)
            Within-region active index for each resource (for :meth:`harvest`).
        """
        grid = self.grid
        pos = np.asarray(pos, dtype=np.float64).reshape(1, grid.D)
        ridx = grid.region_of(pos)
        nbr_ids = grid.neighbour_ids(ridx)[0]            # (3**D,)

        cap = self._cap
        disp_chunks, rid_chunks, slot_chunks = [], [], []
        for rid in nbr_ids:
            base, n = self._region_slice(rid)
            if n == 0:
                continue
            phys = self._corner[rid] + self._local[base : base + n] * grid.L
            disp_chunks.append(grid.displacement(pos, phys))
            rid_chunks.append(np.full(n, rid, dtype=np.int64))
            slot_chunks.append(np.arange(n, dtype=np.int64))
        if not disp_chunks:
            return np.zeros((0, grid.D)), np.zeros(0, np.int64), np.zeros(0, np.int64)
        return (
            np.concatenate(disp_chunks, axis=0),
            np.concatenate(rid_chunks),
            np.concatenate(slot_chunks),
        )

    def harvest(self, region_id, slot):
        """Remove one resource by ``(region_id, slot)``. O(1) swap-delete.

        Returns
        -------
        bool
            True if a resource was removed; False if the slot was already
            empty (e.g. another agent harvested it first this step).
        """
        n = self.count[region_id]
        if not (0 <= slot < n):
            return False
        base = region_id * self._cap
        self._local[base + slot] = self._local[base + n - 1]
        self.count[region_id] = n - 1
        return True

    # ----------------------------------------------------- field protocol
    def sample(self, pos):
        """Nearest-resource displacement for each query position.

        Parameters
        ----------
        pos : ndarray, shape (n, D)
            Physical query positions.

        Returns
        -------
        disp : ndarray, shape (n, D)
            Displacement to the nearest resource (NaN row if none in range).
        dist2 : ndarray, shape (n,)
            Squared distance to that resource (inf if none).

        Notes
        -----
        Searches the Moore neighbourhood of each query position. This is the
        sensing primitive the greedy population uses; it is a per-agent loop
        for now and is a natural ``@njit`` target once the core is validated.
        """
        pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
        n, D = pos.shape
        out_disp = np.full((n, D), np.nan)
        out_d2 = np.full(n, np.inf)
        for i in range(n):
            disp, _, _ = self.neighbourhood(pos[i])
            if disp.shape[0]:
                d2 = np.einsum("ij,ij->i", disp, disp)
                j = int(np.argmin(d2))
                out_disp[i] = disp[j]
                out_d2[i] = d2[j]
        return out_disp, out_d2

    def deposit(self, pos, amount):
        """Resource fields do not accept deposits."""
        raise NotImplementedError("ResourceField does not support deposit()")
