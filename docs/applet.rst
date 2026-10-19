Applet
======

.. raw:: html

   <link rel="stylesheet" href="_static/leafs_applet.css">
   <div id="leafs-app"></div>
   <script src="_static/leafs_applet.js"></script>

The 2d world of pyLEAFS, running live in your browser. It is the world
:meth:`~pyLEAFS.simulation.Simulation.forager` builds, with the same rules and
parameters, applied in the same order: the births and deaths of
:class:`~pyLEAFS.fields.ResourceField`, region by region, then the steering,
motion, harvest, metabolism, budding and death of
:class:`~pyLEAFS.population.Population`, stage by stage. Nothing is
precomputed.

The applet is yours to take apart. It lives in the library's repository as
`docs/_static/leafs_applet.js
<https://github.com/EternalTime/LEAFS/blob/master/docs/_static/leafs_applet.js>`_:
plain JavaScript with no build step and no dependencies. The first part is the
model, ported line for line from the modules it documents, along with the
samplers behind its random numbers. The rest draws the world and its two
panels. All of the model's randomness passes through one function, ``draw``, so
you can feed it the draws of a seeded Python run and compare the two step by
step.

Green dots are resources and pink disks are foragers, each sized by its fuel,
with a short line showing its heading. The world starts where the Python one
does: the field near its equilibrium of
:math:`N_\mathrm{eq} = \Gamma L^2/(\epsilon\gamma)` resources per region, and
five foragers with full tanks, which bud on the first step. The top panel on the
right follows the census, each population divided by its carrying capacity,
:math:`N_A^{\max} = \Gamma V/\mu_0` for the foragers and
:math:`N_R^{\max} = \Gamma\tau_E V/\epsilon` for the resources: the foragers
boom, graze the field down, and settle. Below it the phase plane shows the same
run on log axes, with the equilibrium line
:math:`N_R/N_R^{\max} + N_A/N_A^{\max} = 1` of :doc:`theory` dashed.

The census settles on that line wherever the foragers survive, as long as
:math:`\Xi` stays above about 0.2. At steady state the energy flowing in,
:math:`\Gamma V`, is either burned by the foragers, :math:`\mu_0 N_A`, or lost
when a resource decays, :math:`\gamma\epsilon N_R`, and the sliders move
the point along the line. Toward the bottom of the :math:`\Xi` slider a single
resource is worth most of a tank, and the energy that spills past a full one is
lost too, so the census sits under the line. Narrow the sensing radius toward
:math:`r_\mathrm{col}` and the foragers, nearly blind, find food by bumping
into it; they still get by, but fewer of them, on a fuller field. Speed them up
and they graze the field bare and close in on :math:`N_A^{\max}`. The
homogeneity :math:`\Xi` decides how the same energy is divided: slide it down
for a few rich resources far apart, up for a pasture of lean ones, and watch the
census swing harder in the sparse world.

Tap a forager to follow it. A ring marks it, a faint circle its sensing radius,
and a pink line its recent path, and the readout adds its fuel, age, harvest and
offspring. Tap empty ground to drop in a new forager there, with a full tank and
a random heading, as a click does in the :class:`~pyLEAFS.viewer.Viewer`. In
Python the parameters are fixed when the world is built; here the sliders act on
the running world, and Reset builds a fresh one from their values.
