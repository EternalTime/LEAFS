# pyLEAFS

pyLEAFS is a Python port of LEAFS (Layered Environment with Agents Foraging
Simulator), an agent-based model for studying how energy harvesting drives the
development of sensors and information-processing behavior. The core is
dimension-agnostic (2d or 3d) and built on a struct-of-arrays NumPy layout.

This first version is a single-species greedy forager on a replenishing Poisson
resource field. Pheromone fields, predators, heterogeneous environments, and
neuroevolution are later layers that attach to it. The original MATLAB classes
live in `matlab/`.

## Installation

```bash
git clone https://github.com/EternalTime/LEAFS.git
cd LEAFS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Requires Python 3.8+ (tested on 3.8 through 3.14); numpy and matplotlib are
installed automatically. The virtual environment is what makes `pip` available
and keeps the install out of a system Python that may refuse it, so activate it
in every new terminal. The [Getting Started
guide](https://damiansowinski.com/LEAFS/getting_started.html) is the authority
on installation and carries the same steps.

## Quick start

```python
from pyLEAFS import Simulation

# greedy forager world matching the LEAFS applet parameters
sim = Simulation.forager(seed=0)
sim.run(1000)
print(sim.populations[0].count, "agents alive")
```

Watch it live (spacebar pauses; click empty space to add an agent; click an
agent to inspect it):

```python
from pyLEAFS import Viewer

sim = Simulation.forager(seed=0)
Viewer(sim).play()
```

## Documentation

The docs live at [damiansowinski.com/LEAFS](https://damiansowinski.com/LEAFS/),
or run `import pyLEAFS; pyLEAFS.docs()` to open them. To build them locally:

```bash
source .venv/bin/activate
pip install -e '.[docs]'
make -C docs html
```

The rendered pages land in `docs/_build/html`.

## Tests

```bash
source .venv/bin/activate
pip install -e '.[test]'
pytest
```

## License

MIT
