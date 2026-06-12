"""Forager populations.

A :class:`Population` is a struct-of-arrays bag of agent bodies sharing a
:class:`Grid`. Each agent has a position, a unit heading, a fuel reserve, and
some bookkeeping. The v1 controller is *greedy* and lives in the agent body:
steer toward the nearest resource within sensing range, otherwise move
ballistically with rotational diffusion.

The struct-of-arrays layout (parallel arrays rather than per-agent objects)
keeps the step vectorizable and dimension-agnostic, and is the natural target
for a later Numba pass. The ``Controller`` split (greedy vs. RNN) is reserved
for a later layer; for now ``step`` calls the built-in greedy steering.
"""

import numpy as np

from pyLEAFS.grid import Grid


class Population:
    """A single-species greedy forager population (struct-of-arrays).

    Parameters
    ----------
    grid : Grid
        Shared spatial grid.
    v : float
        Translational speed.
    dt : float
        Timestep.
    r_collect : float
        Collection radius; resources within this distance are harvested.
    R_sense : float
        Sensing radius; only resources within this distance steer the agent.
    mu0 : float
        Basal metabolic rate (fuel burned per unit time).
    s_max : float
        Maximum fuel reserve.
    sigma : float, optional
        Rotational-diffusion noise strength (default 0.1).
    tau_A : float, optional
        Heading persistence timescale (default 1.0).
    repro_fraction : float, optional
        Fuel threshold for budding, as a fraction of ``s_max`` (default 0.8).
    capacity : int, optional
        Initial array capacity; grows automatically as needed (default 256).

    Attributes
    ----------
    count : int
        Number of live agents.
    pos : ndarray, shape (count, D)
        Agent positions (a live view; copy if you need to retain it).
    heading : ndarray, shape (count, D)
        Unit heading vectors.
    fuel : ndarray, shape (count,)
        Fuel reserves.
    age : ndarray, shape (count,)
        Age in timesteps.
    ids : ndarray, shape (count,)
        Stable unique id per agent (survives compaction).
    harvested : ndarray, shape (count,)
        Lifetime resources collected per agent.
    offspring : ndarray, shape (count,)
        Number of daughters budded per agent.
    """

    def __init__(self, grid, v, dt, r_collect, R_sense, mu0, s_max,
                 sigma=0.1, tau_A=1.0, repro_fraction=0.8, capacity=256):
        self.grid = grid
        self.v = float(v)
        self.dt = float(dt)
        self.r_collect = float(r_collect)
        self.R_sense = float(R_sense)
        self.mu0 = float(mu0)
        self.s_max = float(s_max)
        self.sigma = float(sigma)
        self.tau_A = float(tau_A)
        self.repro_threshold = float(repro_fraction) * self.s_max

        D = grid.D
        self._cap = int(capacity)
        self.count = 0
        self._next_id = 0
        self._pos = np.zeros((self._cap, D))
        self._heading = np.zeros((self._cap, D))
        self._fuel = np.zeros(self._cap)
        self._age = np.zeros(self._cap, dtype=np.int64)
        self._ids = np.zeros(self._cap, dtype=np.int64)
        self._harvested = np.zeros(self._cap, dtype=np.int64)
        self._offspring = np.zeros(self._cap, dtype=np.int64)

    # -------------------------------------------------------- live views
    @property
    def pos(self):
        return self._pos[: self.count]

    @property
    def heading(self):
        return self._heading[: self.count]

    @property
    def fuel(self):
        return self._fuel[: self.count]

    @property
    def age(self):
        return self._age[: self.count]

    @property
    def ids(self):
        return self._ids[: self.count]

    @property
    def harvested(self):
        return self._harvested[: self.count]

    @property
    def offspring(self):
        return self._offspring[: self.count]

    # --------------------------------------------------------- spawning
    def _ensure_capacity(self, extra):
        need = self.count + extra
        if need <= self._cap:
            return
        new_cap = max(need, 2 * self._cap)
        def grow(a):
            shape = (new_cap,) + a.shape[1:]
            b = np.zeros(shape, dtype=a.dtype)
            b[: self._cap] = a
            return b
        self._pos = grow(self._pos)
        self._heading = grow(self._heading)
        self._fuel = grow(self._fuel)
        self._age = grow(self._age)
        self._ids = grow(self._ids)
        self._harvested = grow(self._harvested)
        self._offspring = grow(self._offspring)
        self._cap = new_cap

    def add(self, pos, heading=None, fuel=None, rng=None):
        """Add one agent. Returns its stable id.

        Parameters
        ----------
        pos : array_like, shape (D,)
            Physical position (wrapped into the domain).
        heading : array_like, shape (D,), optional
            Heading vector (normalised); random if omitted (requires ``rng``).
        fuel : float, optional
            Initial fuel; defaults to ``s_max``.
        rng : numpy.random.Generator, optional
            Random source, used when ``heading`` is omitted.
        """
        self._ensure_capacity(1)
        i = self.count
        self._pos[i] = self.grid.wrap(np.asarray(pos, dtype=np.float64).reshape(1, self.grid.D))[0]
        if heading is None:
            heading = self._random_headings(1, rng)[0]
        else:
            heading = np.asarray(heading, dtype=np.float64)
            norm = np.linalg.norm(heading)
            heading = heading / norm if norm > 0 else self._random_headings(1, rng)[0]
        self._heading[i] = heading
        self._fuel[i] = self.s_max if fuel is None else float(fuel)
        self._age[i] = 0
        self._ids[i] = self._next_id
        self._harvested[i] = 0
        self._offspring[i] = 0
        self._next_id += 1
        self.count += 1
        return self._ids[i]

    def seed(self, n, rng, fuel=None):
        """Add ``n`` agents at uniformly random positions and headings."""
        pos = rng.random((n, self.grid.D)) * self.grid.extent
        head = self._random_headings(n, rng)
        for k in range(n):
            self.add(pos[k], head[k], fuel=fuel)

    def _random_headings(self, n, rng):
        """Draw ``n`` unit vectors uniformly on the sphere/circle."""
        if rng is None:
            rng = np.random.default_rng()
        v = rng.standard_normal((n, self.grid.D))
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return v / norm

    def index_of(self, agent_id):
        """Return the current array index of ``agent_id``, or None if dead."""
        match = np.nonzero(self._ids[: self.count] == agent_id)[0]
        return int(match[0]) if match.size else None

    # ------------------------------------------------------------- step
    def step(self, fields, rng):
        """Advance every agent one timestep against the field list.

        Greedy steering, motion, harvesting against the first
        :class:`ResourceField` in ``fields``, metabolism, budding, and death.
        Returns nothing; mutates the population in place.
        """
        if self.count == 0:
            return
        resource = self._first_resource_field(fields)

        # --- greedy steering: aim heading at nearest resource in range ---
        if resource is not None:
            disp, dist2 = resource.sample(self.pos)
            sensed = dist2 <= self.R_sense ** 2
            if np.any(sensed):
                d = disp[sensed]
                norm = np.linalg.norm(d, axis=1, keepdims=True)
                norm[norm == 0] = 1.0
                self._heading[: self.count][sensed] = d / norm

        # --- rotational diffusion on unsensed/all agents ---
        self._diffuse_heading(rng)

        # --- translate (toroidal) ---
        self._pos[: self.count] = self.grid.wrap(
            self.pos + self.v * self.heading * self.dt
        )
        self._age[: self.count] += 1

        # --- harvest + metabolism ---
        harvest_energy = np.zeros(self.count)
        if resource is not None:
            harvest_energy = self._harvest(resource)
        self._fuel[: self.count] = np.minimum(
            self.s_max, self.fuel - self.mu0 * self.dt + harvest_energy
        )

        # --- reproduce then cull (order matches the applet) ---
        self._reproduce(rng)
        self._cull()

    def _first_resource_field(self, fields):
        from pyLEAFS.fields import ResourceField
        for f in fields:
            if isinstance(f, ResourceField):
                return f
        return None

    def _diffuse_heading(self, rng):
        """Rotate each heading by Gaussian angular noise (sigma / sqrt(tau_A))."""
        n, D = self.count, self.grid.D
        scale = self.sigma / np.sqrt(self.tau_A) * np.sqrt(self.dt)
        if D == 2:
            theta = np.arctan2(self.heading[:, 1], self.heading[:, 0])
            theta = theta + scale * rng.standard_normal(n)
            self._heading[:n, 0] = np.cos(theta)
            self._heading[:n, 1] = np.sin(theta)
        else:
            # add a small random tangential kick, then renormalise
            noise = scale * rng.standard_normal((n, D))
            h = self.heading + noise
            norm = np.linalg.norm(h, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            self._heading[:n] = h / norm

    def _harvest(self, resource):
        """Collect resources within r_collect; return per-agent energy gained."""
        energy = np.zeros(self.count)
        r2 = self.r_collect ** 2
        for i in range(self.count):
            disp, rids, slots = resource.neighbourhood(self.pos[i])
            if disp.shape[0] == 0:
                continue
            d2 = np.einsum("ij,ij->i", disp, disp)
            hits = np.nonzero(d2 <= r2)[0]
            for j in hits:
                if resource.harvest(int(rids[j]), int(slots[j])):
                    energy[i] += resource.epsilon
                    self._harvested[i] += 1
        return energy

    def _reproduce(self, rng):
        """Bud daughters from agents at or above the fuel threshold."""
        ready = np.nonzero(self.fuel[: self.count] >= self.repro_threshold)[0]
        if ready.size == 0:
            return
        offset = self.r_collect
        for i in ready:
            half = self._fuel[i] / 2.0
            self._fuel[i] = half
            self._offspring[i] += 1
            direction = self._random_headings(1, rng)[0]
            daughter_pos = self.grid.wrap((self.pos[i] + offset * direction).reshape(1, self.grid.D))[0]
            self.add(daughter_pos, heading=direction, fuel=half, rng=rng)

    def _cull(self):
        """Swap-compact out agents with non-positive fuel."""
        alive = self._fuel[: self.count] > 0
        if np.all(alive):
            return
        keep = np.nonzero(alive)[0]
        nk = keep.size
        for a in (self._pos, self._heading, self._fuel, self._age,
                  self._ids, self._harvested, self._offspring):
            a[:nk] = a[: self.count][keep]
        self.count = nk
