pyLEAFS
=======

An organism that must eat to live is, whether it knows it or not, in the
business of information. To harvest energy from a structured environment it has
to find that energy, and to find it reliably it has to build something that
tells it where the energy is: a sensor. LEAFS --- the Layered Environment with
Agents Foraging Simulator --- is a model for studying that pressure directly.
Agents forage a replenishing field of resources; energy is the only currency;
and the central question is how the demand for energy shapes the sensors and
information-processing behavior the agents come to rely on.

The model is built in layers. At the bottom is a *resource layer*: a stochastic
field of energy quanta, born and decaying as a Poisson point process, tuned by a
single homogeneity parameter :math:`\Xi` that slides the world from sparse and
patchy to dense and uniform. Above it sit *thermodynamic and signalling layers*
--- metabolism, and later pheromone and chemical fields --- through which agents
spend and sense energy. At the top is the *agent layer*: bodies that move,
metabolize, reproduce, and (in later versions) carry evolvable neural
controllers and sensor arrays. Each layer is an addition to the one below, and
the code is organized so that the layers stay separable.

This is the base. The first version implements the bottom of the stack: a
single-species *greedy* forager on the resource field, with no neural
controller, no genome, and no evolution. It reproduces the dynamics of the
``forager`` web applet and the resource-gathering model of the papers below. The
neuroevolution, the heterogeneous environments, the predators, and the
information-theoretic measures are layers still to be added; the core
(:doc:`guide_core`) is structured to receive them.

The science behind the model is developed in three papers:

- Sowinski, Carroll-Nellenback, Markwick, Pinero, Gleiser, Kolchinsky, Ghoshal,
  Frank, *Semantic Information in a Model of Resource Gathering Agents*, PRX Life
  **1**, 023003 (2023). The semantic-information analysis of the forager.
- Sowinski, *Energy Harvesting, Information Architectures, and the Development of
  Senses* (notes). The theoretical foundation: the homogeneity parameter, the
  equations of motion, sensors framed as metabolism-modulating behaviors.
- Sowinski, *Behavioral Convergence in the Neuroevolution of Foraging Recurrent
  Neural Nets driven by Environmental Homogeneity* (preprint). The
  neuroevolution layer that later versions of pyLEAFS will implement.

If you are new here, start with :doc:`getting_started`, then read
:doc:`theory` for the model the code implements and :doc:`guide_core` for the
architecture and how new layers attach.

Guide
^^^^^

.. toctree::
   :maxdepth: 1

   getting_started
   theory
   guide_core

Reference
^^^^^^^^^

.. toctree::
   :maxdepth: 2

   api/pyLEAFS
   license

Citing
^^^^^^

Sowinski DR. *pyLEAFS* [computer software]. Version 0.1.0. 2026. Accessed July 21, 2026. https://github.com/EternalTime/LEAFS

.. code-block:: bibtex

   @software{sowinski_pyleafs,
     author  = {Sowinski, Damian R.},
     title   = {pyLEAFS: Layered Environment with Agents Foraging Simulator},
     year    = {2026},
     version = {0.1.0},
     url     = {https://github.com/EternalTime/LEAFS}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
