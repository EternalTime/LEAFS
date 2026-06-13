The core and its seams
======================

pyLEAFS is a base meant to grow. This page describes the architecture of the v1
core and, more importantly, the *seams* it leaves open so that each later layer
attaches as an addition rather than a rewrite. If you are extending pyLEAFS,
read this first.

Three design commitments
------------------------

**Dimension-agnostic.** There are no separate 2d and 3d codebases. Dimension is
a property of the :class:`~pyLEAFS.Grid`, set by the length of its ``shape``, and
positions are always ``(n, D)`` arrays. Code that avoids hard-coding ``D`` works
in both.

**Struct-of-arrays.** Resources and agents are stored as parallel NumPy arrays,
not as lists of objects. A population's positions are one ``(n, D)`` array, its
fuel one ``(n,)`` array, and so on. This keeps the per-step work vectorizable and
dimension-agnostic, and is the natural target for a later Numba pass.

**Numba-ready, not yet Numba.** Hot loops (per-agent sensing and harvesting) are
written as plain Python/NumPy so the core can be validated for correctness
first. Decorating them with ``@njit`` is deferred until the physics is pinned.

The shared grid
---------------

Every field and every population references one :class:`~pyLEAFS.Grid`. It owns
the geometry: the region partition, the toroidal wrap, the minimum-image
displacement, and the Moore-neighbour lookup (:math:`3^{D}` neighbours per
region). Defining geometry in exactly one place is what lets fields and
populations interoperate without each re-deriving coordinate conventions.

The field list
--------------

A *field* is anything spatially distributed that the simulation steps and agents
sample. :class:`~pyLEAFS.Field` is the structural protocol --- ``step``,
``sample``, ``deposit`` --- and :class:`~pyLEAFS.ResourceField` is the only
implementation in v1.

The key seam is that :class:`~pyLEAFS.Simulation` holds a **list** of fields and
steps each one:

.. code-block:: python

   for f in self.fields:
       f.step(self.rng)

A pheromone field, a chemical gradient, or a temperature field is a new class
satisfying the protocol and a new entry in that list. The ``deposit`` verb exists
precisely so a pheromone field --- which agents write to as well as read --- slots
in without changing any call site. Nothing about the step loop assumes a single
hard-coded resource field.

The spatial hash
----------------

:class:`~pyLEAFS.SpatialHash` buckets an arbitrary ``(n, D)`` point set into grid
regions and answers neighbourhood queries. It is deliberately agnostic about what
the points are. The resource field keeps its own internal buffer, but predators
sensing prey, agents sensing each other, or any future point-set interaction can
share this one index rather than each growing a bespoke one.

The population list
-------------------

Like fields, populations are a **list** that the step loop iterates:

.. code-block:: python

   for p in self.populations:
       p.step(self.fields, self.rng)

v1 has one greedy-forager population, but the loop is written for many. Predators
are a second population whose ``step`` consumes the first; a second forager
species is a third. The ordering of the list is the interaction order, which is
where multi-trophic dynamics will be expressed.

Each agent carries a stable ``id`` that survives the swap-compaction used when
agents die, so a caller (the viewer's inspector, a lineage tracker) can follow a
specific agent across steps.

The reserved controller
-----------------------

The one seam that is named but not yet abstracted is the **controller**: the map
from what an agent senses to how it steers. In v1 this map is the greedy rule,
and it lives directly in the agent body (``Population.step``). The later
neuroevolution layer splits it out into a ``Controller`` --- a ``GreedyController``
and an ``RNNController`` (an Elman network with an evolvable genome and sensor
array) --- selected per population or per agent. The greedy logic is kept
self-contained so that this extraction is a refactor of one method, not a
redesign.

Adding a layer
--------------

Concretely, to add a new model layer you will typically:

1. Write a new :class:`~pyLEAFS.Field` (e.g. a pheromone field) or a new
   ``Population`` (e.g. predators), reusing the shared :class:`~pyLEAFS.Grid` and,
   where useful, :class:`~pyLEAFS.SpatialHash`.
2. Append it to the simulation's ``fields`` or ``populations`` list.
3. Add a factory or extend :meth:`~pyLEAFS.Simulation.forager` to assemble the
   new world, and pin its behavior with a test.

Because the loops already iterate lists and the geometry already lives in the
grid, the existing layers do not change.
