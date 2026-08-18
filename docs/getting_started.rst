Getting started
===============

Installation
------------

.. code-block:: bash

   git clone https://github.com/EternalTime/LEAFS.git
   cd LEAFS
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .

pyLEAFS requires Python 3.8+ (tested on 3.8 through 3.14); numpy and matplotlib
are installed automatically. The virtual environment is what makes ``pip``
available and keeps the install out of a system Python that may refuse it;
activate it in every new terminal before running any of the commands below.

Building these docs
-------------------

.. code-block:: bash

   source .venv/bin/activate
   pip install -e '.[docs]'
   make -C docs html

The rendered pages land in ``docs/_build/html``.

A first simulation
-------------------

The :meth:`~pyLEAFS.Simulation.forager` factory builds the v1 world --- one
resource field, one greedy-forager population --- with the parameters of the
``forager`` applet:

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

The :class:`~pyLEAFS.Viewer` opens an interactive matplotlib window:

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
there. The viewer targets two-dimensional worlds.

The homogeneity knob
---------------------

The environment is controlled by a single dimensionless parameter, the
homogeneity :math:`\Xi`. It sets how patchy or uniform the resource field is by
fixing the energy per resource:

.. code-block:: python

   sparse = Simulation.forager(seed=0, Xi=0.3)   # patchy: few, rich resources
   dense  = Simulation.forager(seed=0, Xi=1.0)   # uniform: many, lean resources

   print(sparse.fields[0].N_eq)   # equilibrium resources per region
   print(dense.fields[0].N_eq)

Larger :math:`\Xi` means a more homogeneous world with more, lower-energy
resources. See :doc:`theory` for the definition and its role in the model.

Two or three dimensions
-----------------------

The core is dimension-agnostic. The length of the grid ``shape`` selects the
dimension:

.. code-block:: python

   flat = Simulation.forager(seed=0, shape=(10, 10))      # 2d
   solid = Simulation.forager(seed=0, shape=(10, 10, 10))  # 3d

   solid.run(500)

The :class:`~pyLEAFS.Viewer` renders 2d worlds; 3d runs are headless for now.

Reproducibility
---------------

Every stochastic part of the simulation draws from a single
``numpy.random.Generator`` threaded through from the seed. Two runs built with
the same ``seed`` produce identical histories:

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
