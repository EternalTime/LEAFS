"""pyLEAFS: Layered Environment with Agents Foraging Simulator.

A dimension-agnostic (2d or 3d) agent-based foraging model built on a
struct-of-arrays NumPy layout. The first version is a single-species greedy
forager on a replenishing Poisson resource field. Pheromone fields, predators,
heterogeneous environments, and neuroevolution are later layers that attach to
that core.

Modules
-------
grid
    Shared Grid: region partition, toroidal Moore neighbours, coordinate maps.
fields
    Field protocol and ResourceField (Poisson birth / Binomial death).
spatialhash
    Generic SpatialHash over any (n, D) point set.
population
    Greedy forager Population (struct-of-arrays bodies, metabolism, budding).
simulation
    Multi-population step loop and parameter factories.
viewer
    Interactive matplotlib viewer (pause, add agents, inspect).
palette
    Named RGB colours for LEAFS visualisation.
"""

def docs():
    """Open the online pyLEAFS documentation in a web browser."""
    import webbrowser
    webbrowser.open('https://damiansowinski.com/LEAFS/')


from pyLEAFS.grid import Grid
from pyLEAFS.fields import Field, ResourceField
from pyLEAFS.spatialhash import SpatialHash
from pyLEAFS.population import Population
from pyLEAFS.simulation import Simulation
from pyLEAFS.viewer import Viewer
from pyLEAFS import palette

__all__ = [
    "Grid",
    "Field",
    "ResourceField",
    "SpatialHash",
    "Population",
    "Simulation",
    "Viewer",
    "palette",
    "docs",
]
