# pyLEAFS

pyLEAFS is a Python port of LEAFS (Layered Environment with Agents Foraging
Simulator), an agent-based model for studying how energy harvesting drives the
development of sensors and information-processing behavior. The core is
dimension-agnostic (2d or 3d) and built on a struct-of-arrays NumPy layout.

This first version is a single-species greedy forager on a replenishing Poisson
resource field. Later layers (pheromone fields, predators, heterogeneous
environments, neuroevolution) are add-ons enabled by the extension seams in the
core.

The original MATLAB classes live in `matlab/`.

## Installation

```bash
cd pyLEAFS
pip install -e .
```

Requires Python 3.8+; numpy and matplotlib are installed automatically.

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

## Tests

```bash
pip install pytest
pytest
```

## License

MIT
