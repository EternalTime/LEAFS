Architecture
============

pyLEAFS v1 is one resource field and one greedy-forager population, wired so
that later layers attach as additions rather than rewrites. Read this before
extending it.

Conventions
-----------

**Dimension lives in the grid.** Dimension is a property of the
:class:`~pyLEAFS.Grid`, set by the length of its ``shape``, and positions are
always ``(n, D)`` arrays. There is no separate 2d and 3d codebase; code that
avoids hard-coding ``D`` works in both - the energy per resource, for one, is
derived from the homogeneity with ``D`` as the exponent (see :doc:`theory`).
Two places do dispatch on ``D``, because the geometry there genuinely differs:
:class:`~pyLEAFS.viewer.Viewer`, since a flat box and a rotatable one are
different matplotlib axes, and heading diffusion in
:class:`~pyLEAFS.Population`, since a 2d heading rotates by an angle where a 3d
one takes a renormalized tangential kick. Both branch behind a single class
rather than splitting into two.

**Arrays, not objects.** Resources and agents are stored as parallel NumPy
arrays - a population's positions are one ``(n, D)`` array, its fuel one
``(n,)`` array - which keeps the per-step work vectorizable and
dimension-agnostic, and is the natural target for a later Numba pass.

**Plain NumPy for now.** The hot loops, per-agent sensing and harvesting, are
ordinary Python so the core can be validated for correctness first. ``@njit``
waits until the physics is pinned.

The grid
--------

Every field and every population references one :class:`~pyLEAFS.Grid`, which
owns the geometry: the region partition, the toroidal wrap, the minimum-image
displacement, and the Moore-neighbour lookup (:math:`3^{D}` neighbours per
region). Defining it once is what lets fields and populations interoperate
without each re-deriving coordinate conventions.

The field list
--------------

A *field* is anything spatially distributed that the simulation steps and agents
sample. :class:`~pyLEAFS.Field` is the structural protocol - ``step``,
``sample``, ``deposit`` - and :class:`~pyLEAFS.ResourceField` is its only
implementation in v1. :class:`~pyLEAFS.Simulation` holds a *list* of fields and
steps each one:

.. code-block:: python

   for f in self.fields:
       f.step(self.rng)

Any other field - pheromone, chemical, thermal - is a new class satisfying the
protocol and a new entry in that list. The ``deposit`` verb is there so that a
pheromone field, which agents write to as well as read, slots in without changing
a call site.

The spatial hash
----------------

:class:`~pyLEAFS.SpatialHash` buckets an arbitrary ``(n, D)`` point set into grid
regions and answers neighbourhood queries, agnostic about what the points are.
The resource field keeps its own buffer, but predators sensing prey, agents
sensing each other, and any future point-set interaction can share this one index.

The population list
-------------------

Populations are likewise a *list* that the step loop iterates:

.. code-block:: python

   for p in self.populations:
       p.step(self.fields, self.rng)

v1 has one greedy-forager population, but the loop is written for many. Predators
are a second population whose ``step`` consumes the first; a second forager
species is a third. List order is interaction order, which is where
multi-trophic dynamics will be expressed.

Each agent carries a stable ``id`` that survives the swap-compaction used when
agents die, so a caller - the viewer's inspector, a lineage tracker - can follow
one agent across steps.

The controller
--------------

The map from what an agent senses to how it steers is not yet abstracted: in v1
it is the greedy rule, living in ``Population.step``. The neuroevolution layer
splits it into a ``Controller``: a ``GreedyController`` and an
``RNNController`` - an Elman network with an evolvable genome and sensor
array - chosen per population or per agent. Keeping the greedy logic
self-contained makes that extraction a refactor of one method, not a redesign.

Adding a layer
--------------

Write a new :class:`~pyLEAFS.Field` (a pheromone field, say) or a new
``Population`` (predators), reusing the shared :class:`~pyLEAFS.Grid` and, where
useful, :class:`~pyLEAFS.SpatialHash`; append it to the simulation's ``fields``
or ``populations`` list; then add a factory, or extend
:meth:`~pyLEAFS.Simulation.forager`, to assemble the new world and pin its
behavior with a test. The loops already iterate lists and the geometry already
lives in the grid, so the existing layers do not change.
