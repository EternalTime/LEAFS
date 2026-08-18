The model
=========

pyLEAFS v1 implements a replenishing resource field and a single greedy forager
living on it. What follows states the model rather than derives it; the
derivations and the notation are in the papers on the :doc:`index`. Sections
marked *(later layer)* name extensions not yet implemented.

The resource field
------------------

Resources are energy quanta scattered through a :math:`D`-dimensional toroidal
box as a Poisson point process. In each region of the grid, one timestep of
duration :math:`dt` births and kills them stochastically:

.. math::

   k_\mathrm{grow}  &\sim \mathrm{Poisson}\!\left(\frac{\Gamma\,L^{D}\,dt}{\epsilon}\right) \\
   k_\mathrm{decay} &\sim \mathrm{Binomial}\!\left(N,\; 1 - e^{-\gamma\,dt}\right)

where :math:`\Gamma` is the energy influx per unit volume, :math:`\gamma` the
resource decay rate, :math:`\epsilon` the energy carried by one resource, and
:math:`L` the region side length. Births place a resource uniformly in the
region and deaths remove uniformly chosen ones; both counts come from two
vectorized calls per step, with the grow/decay order randomized per region to
avoid bias. The ``ResourceField`` is the struct-of-arrays descendant of the
MATLAB ``Region``/``Environment`` pair.

With no agents present the field relaxes to an equilibrium count per region

.. math::

   N_\mathrm{eq} = \frac{\Gamma\,L^{D}}{\epsilon\,\gamma},

which :class:`~pyLEAFS.ResourceField` exposes as ``N_eq``.

Homogeneity
-----------

Rather than set :math:`\epsilon` directly, the model is parameterized by a single
dimensionless **homogeneity** :math:`\Xi`, the ratio of the agent length scale
:math:`\ell_A` to the mean resource spacing :math:`\ell_E`:

.. math::

   \Xi = \frac{\ell_A}{\ell_E},
   \qquad
   \ell_E = \left(\frac{\epsilon}{\Gamma\,\tau_E}\right)^{1/D},
   \qquad
   \tau_E = 1/\gamma .

Taking the agent length scale to be the collection radius
(:math:`\ell_A = r_\mathrm{col}` in the 2d applet) and inverting for the energy
per resource gives the relation :meth:`~pyLEAFS.Simulation.forager` uses:

.. math::

   \epsilon = \frac{\Gamma}{\,(\Xi / r_\mathrm{col})^{D}\,\gamma\,}.

Small :math:`\Xi` is a *heterogeneous* world: few energy-rich resources, widely
spaced. Large :math:`\Xi` is a *homogeneous* one: many lean resources, densely
packed. Homogeneity is the central control variable of the whole research
program: the semantic-information threshold, the foraging phenotypes, and the
punctuated evolutionary dynamics are all studied as functions of :math:`\Xi`.

The agent
---------

An agent is a body with a position :math:`\mathbf{x}`, a unit heading
:math:`\hat{\mathbf{n}}`, and a fuel reserve :math:`s`. Each timestep it senses,
steers, moves, harvests, and metabolizes.

**Sensing and steering.** The v1 controller is *greedy*: the agent looks for the
nearest resource within a sensing radius :math:`R_\mathrm{sense}` and turns to
face it. With nothing in range its heading diffuses,

.. math::

   \hat{\mathbf{n}} \leftarrow R\!\left(\frac{\sigma}{\sqrt{\tau_A}}\sqrt{dt}\;\xi\right)\hat{\mathbf{n}},
   \qquad \xi \sim \mathcal{N}(0, 1),

a rotation by Gaussian angular noise of strength :math:`\sigma/\sqrt{\tau_A}`,
realized in 2d as a heading-angle perturbation and in 3d as a renormalized
tangential kick. The greedy rule is the perfect-fidelity limit of the sensor
model; replacing it with a learned controller is a *(later layer)*.

**Motion.** The agent advances ballistically and wraps on the torus:

.. math::

   \mathbf{x} \leftarrow \big(\mathbf{x} + v\,\hat{\mathbf{n}}\,dt\big) \bmod \mathbf{extent},

with :math:`v` the translational speed.

**Harvesting and metabolism.** Every resource within the collection radius
:math:`r_\mathrm{col}` is consumed, each returning energy :math:`\epsilon`. Fuel
follows

.. math::

   \frac{ds}{dt} = -\mu_0 + h,

where :math:`\mu_0` is the basal metabolic rate and :math:`h` the harvest rate,
capped at :math:`s \le s_\mathrm{max}`. Metabolism here is a single basal term;
the *(later layer)* in which sensors and actions modulate :math:`\mu_0` - sensors
as behaviors that cost energy - is the subject of the energy-harvesting notes on
the :doc:`index`.

**Reproduction and death.** When fuel reaches the threshold
:math:`s_\mathrm{rep} = 0.8\,s_\mathrm{max}` the agent *buds*: it spawns a
daughter a short distance away and the two split the fuel equally. An agent dies
when :math:`s \le 0`. In v1 the daughter is an exact clone; mutation of a genome
is a *(later layer)*.

Population dynamics
-------------------

With agents drawing down the field, the resource and agent counts trace a
predator--prey-like relation framed by two carrying capacities:

.. math::

   N_R^{\max} = \frac{\Gamma\,\tau_E\,V}{\epsilon}, \qquad
   N_A^{\max} = \frac{\Gamma\,V}{\mu_0},

with :math:`V` the domain volume. :math:`N_R^{\max}` is the no-agent resource
equilibrium and varies with :math:`\Xi`; :math:`N_A^{\max}` is the agent carrying
capacity and does not. At equilibrium the normalized counts lie on a line,

.. math::

   \frac{N_R}{N_R^{\max}} + \frac{N_A}{N_A^{\max}} = 1,

the diagonal of the phase plane. Starting from the empty-agent corner, a seeded
population overshoots, grazes the field down, and settles into a sustained
oscillation about this line rather than going extinct; that is the behavior the
v1 test suite pins. Plotting the trajectory in the
:math:`(N_R/N_R^{\max},\,N_A/N_A^{\max})` plane, and detecting the transition
onto the equilibrium line, is the first of the *(later)* observables.

Semantic information
--------------------

The point of building the model around energy and sensing is to measure the
*information* an agent needs to stay alive. In the Kolchinsky--Wolpert framework
the **semantic information** is the share of the agent--environment mutual
information that is actually necessary for viability, the expected lifetime.
Scramble the sensor past a critical noise level :math:`\eta_c` and viability
collapses; below it, noise barely matters. The threshold depends only on the
agent's geometry,

.. math::

   \sin(\pi\,\eta_c) = \frac{r}{R},

with :math:`r` the collection radius and :math:`R` the sensing radius. Computing
the mutual information, transfer entropy, and viability curve from logged
trajectories is a planned analysis *(later layer)*; v1 provides the dynamics they
are measured on.

Roadmap of layers
-----------------

Everything marked *(later layer)* attaches to the v1 core described in
:doc:`guide_core`: a pheromone field (reaction--diffusion) as a second entry in
the field list, predators and trophic levels as additional populations in the
step loop, a heterogeneous environment with spatially varying :math:`\Gamma`,
neuroevolution replacing the greedy rule with an evolvable recurrent controller
and sensor array under mutation and selection, a chemoton with internal
metabolism in place of the single basal term, and the information-theoretic
observables: semantic information, mutual information, and transfer entropy.
