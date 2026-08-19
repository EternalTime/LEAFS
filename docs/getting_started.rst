Getting started
===============

Installation
------------

.. code-block:: bash

   git clone https://github.com/EternalTime/LEAFS.git
   cd LEAFS
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e .

pyLEAFS requires Python 3.8+ (tested on 3.8 through 3.14); numpy and matplotlib
are installed automatically. Activate the virtual environment in every new
terminal: it makes ``pip`` available and keeps the install out of a system
Python that may refuse it.

Building these docs
-------------------

.. code-block:: bash

   source .venv/bin/activate
   pip install -e '.[docs]'
   make -C docs html

The rendered pages land in ``docs/_build/html``.

A first simulation
-------------------

:meth:`~pyLEAFS.Simulation.forager` builds the v1 world - one resource field,
one greedy-forager population - with the parameters of the ``forager`` applet:

.. code-block:: python

   from pyLEAFS import Simulation

   sim = Simulation.forager(seed=0)
   sim.run(1000)
   print(sim.populations[0].count, "agents alive")
   print(sim.fields[0].total(), "resources")

``run`` stops early if the population goes extinct; pass
``stop_on_extinction=False`` to advance a fixed number of steps regardless.

Watching it live
----------------

:class:`~pyLEAFS.Viewer` opens an interactive matplotlib window:

.. code-block:: python

   from pyLEAFS import Simulation, Viewer

   Viewer(Simulation.forager(seed=0)).play()

Controls:

==================  ========================================================
spacebar            Pause / resume.
click empty space   Add a new agent at the cursor (works paused or running).
click on an agent   Select it: a ring appears, the side panel shows its
                    state (fuel, age, harvested count, offspring, heading),
                    and its recent trajectory is drawn as a trail.
==================  ========================================================

Clicking empty space while an agent is selected deselects it and adds an agent
there. The same controls drive a 3d world; see below for what a click means
once the box can be rotated.

The homogeneity knob
---------------------

A single dimensionless parameter controls the environment. The homogeneity
:math:`\Xi` fixes the energy per resource, and with it how patchy or uniform
the field is:

.. code-block:: python

   sparse = Simulation.forager(seed=0, Xi=0.3)   # patchy: few, rich resources
   dense  = Simulation.forager(seed=0, Xi=1.0)   # uniform: many, lean resources

   print(sparse.fields[0].N_eq)   # equilibrium resources per region
   print(dense.fields[0].N_eq)

Larger :math:`\Xi` means a more homogeneous world with more, lower-energy
resources. See :doc:`theory` for the definition and its role.

Two or three dimensions
-----------------------

The core is dimension-agnostic, and the length of the grid ``shape`` selects the
dimension:

.. code-block:: python

   flat = Simulation.forager(seed=0, shape=(10, 10))      # 2d
   solid = Simulation.forager(seed=0, shape=(10, 10, 10))  # 3d

   solid.run(500)

:class:`~pyLEAFS.Viewer` follows suit: it reads the dimension from the grid and
builds a flat or a rotatable box, with the same keys, the same side panel, and
the same colours either way.

.. code-block:: python

   from pyLEAFS import Simulation, Viewer

   Viewer(Simulation.forager(seed=0, shape=(10, 10, 10))).play()

A 3d axes already spends left-drag on rotating the view, and a click on the
screen is a line into the picture rather than a single point, so the mouse
means this:

- Left-drag rotates the view, and rotating never selects or adds anything.
- A click that leaves the view where it was - press and release at effectively
  the same place - is a pick.
- A pick selects the agent nearest that line of sight, as long as it is within
  the same ``select_radius`` tolerance as in 2d, measured on screen.
- A pick that lands near no agent adds one on that line, at the depth of the
  centre of the world box.

A 3d world holds far more resources than a 2d one of the same width, so the
default ``shape=(10, 10, 10)`` runs slower per frame than the 2d default; a
smaller ``shape`` or a smaller ``Xi`` keeps it brisk.

Reproducibility
---------------

One ``numpy.random.Generator``, threaded through from the seed, feeds every
stochastic part of the simulation, so two runs with the same ``seed`` produce
identical histories:

.. code-block:: python

   a = Simulation.forager(seed=7); a.run(300, stop_on_extinction=False)
   b = Simulation.forager(seed=7); b.run(300, stop_on_extinction=False)
   assert a.populations[0].count == b.populations[0].count

Tests
-----

.. code-block:: bash

   source .venv/bin/activate
   pip install -e '.[test]'
   pytest
