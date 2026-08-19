"""The simulation: a grid, a list of fields, a list of populations, a clock.

:class:`Simulation` owns the shared :class:`Grid`, a *list* of fields, and a
*list* of populations, and advances them together. Both are lists from the
start so that adding a pheromone field or a predator population later is an
append, not a rewrite. v1 has one resource field and one greedy forager
population; the :meth:`forager` factory builds that world from the LEAFS applet
parameters.
"""

import numpy as np

from pyLEAFS.grid import Grid
from pyLEAFS.fields import ResourceField
from pyLEAFS.population import Population


class Simulation:
    """A clocked collection of fields and populations on a shared grid.

    Parameters
    ----------
    grid : Grid
        Shared spatial grid.
    fields : list
        Fields to step each timestep (e.g. a :class:`ResourceField`).
    populations : list of Population
        Populations to step each timestep.
    dt : float
        Timestep.
    rng : numpy.random.Generator, optional
        Random source; defaults to a fresh unseeded generator.

    Attributes
    ----------
    step_count : int
        Number of timesteps advanced.
    time : float
        Physical time elapsed (``step_count * dt``).

    Examples
    --------
    >>> sim = Simulation.forager(seed=0)
    >>> sim.run(50)
    >>> sim.populations[0].count >= 0
    True
    """

    def __init__(self, grid, fields, populations, dt, rng=None):
        self.grid = grid
        self.fields = list(fields)
        self.populations = list(populations)
        self.dt = float(dt)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.step_count = 0

    @property
    def time(self):
        return self.step_count * self.dt

    def step(self):
        """Advance every field then every population by one timestep."""
        for f in self.fields:
            f.step(self.rng)
        for p in self.populations:
            p.step(self.fields, self.rng)
        self.step_count += 1

    def run(self, n_steps, stop_on_extinction=True):
        """Advance ``n_steps`` timesteps.

        Parameters
        ----------
        n_steps : int
            Number of steps to advance.
        stop_on_extinction : bool, optional
            Stop early if every population is empty (default True).

        Returns
        -------
        int
            Number of steps actually taken.
        """
        for k in range(n_steps):
            self.step()
            if stop_on_extinction and all(p.count == 0 for p in self.populations):
                return k + 1
        return n_steps

    # ----------------------------------------------------------- factory
    @classmethod
    def forager(cls, seed=None, shape=(10, 10), n_agents=5, Xi=0.5):
        """Build the v1 greedy-forager world from ``forager_applet.js``.

        The environment homogeneity ``Xi`` is the primary control: the energy
        per resource is derived from it as in the applet,

        .. math::

            \\epsilon = \\Gamma / \\big((\\Xi / r_\\mathrm{col})^2\\,\\gamma\\big),

        so that the equilibrium resource density rises as ``Xi`` rises (a more
        homogeneous field of many lean resources). Remaining parameters match
        the applet: ``mu0=0.1``, ``dt=0.02``, ``v=20``, ``R_sense=6``,
        ``r_collect=1``, ``s_max=1``, reproduction at ``0.8`` with the daughter
        budded ``r_collect`` away.

        Parameters
        ----------
        seed : int, optional
            RNG seed for reproducibility.
        shape : tuple of int, optional
            Regions per axis; length 2 or 3 selects the dimension
            (default ``(10, 10)``, the applet's 100x100 domain at L=10).
        n_agents : int, optional
            Initial number of foragers (default 5).
        Xi : float, optional
            Environment homogeneity parameter (default 0.5).

        Returns
        -------
        Simulation
            A seeded simulation with one resource field and one forager
            population, both populated near their initial conditions.
        """
        rng = np.random.default_rng(seed)
        L, dt = 10.0, 0.02
        Gamma, gamma, r_collect = 0.001, 0.1, 1.0
        epsilon = Gamma / ((Xi / r_collect) ** 2 * gamma)
        grid = Grid(shape, L=L)
        resource = ResourceField(grid, Gamma=Gamma, gamma=gamma, epsilon=epsilon, dt=dt)
        resource.seed(rng)
        pop = Population(
            grid, v=20.0, dt=dt, r_collect=r_collect, R_sense=6.0,
            mu0=0.1, s_max=1.0, sigma=0.1, tau_A=1.0, repro_fraction=0.8,
        )
        pop.seed(n_agents, rng)
        return cls(grid, [resource], [pop], dt=dt, rng=rng)
