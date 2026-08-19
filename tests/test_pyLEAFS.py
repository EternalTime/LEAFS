"""Tests pinning the physics and bookkeeping of pyLEAFS v1."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.animation import FuncAnimation                      # noqa: E402

from pyLEAFS import (Grid, ResourceField, SpatialHash, Population,  # noqa: E402
                     Simulation, Viewer)


# ----------------------------------------------------------------- Grid
def test_grid_dimension_from_shape():
    assert Grid((5, 5)).D == 2
    assert Grid((4, 4, 4)).D == 3


def test_grid_region_of_and_wrap():
    g = Grid((5, 5), L=10.0)
    assert np.array_equal(g.region_of(np.array([[12.0, 3.0]])), [[1, 0]])
    # wrapping past the far edge returns to region 0
    assert np.array_equal(g.region_of(np.array([[50.5, 0.0]])), [[0, 0]])


def test_grid_minimum_image_displacement():
    g = Grid((5, 5), L=10.0)             # extent 50
    a = np.array([[1.0, 1.0]])
    b = np.array([[49.0, 1.0]])
    d = g.displacement(a, b)            # should wrap to -2, not +48
    assert np.allclose(d, [[-2.0, 0.0]])


def test_grid_neighbour_count():
    assert Grid((5, 5)).neighbour_ids(np.array([[2, 2]])).shape == (1, 9)
    assert Grid((4, 4, 4)).neighbour_ids(np.array([[1, 1, 1]])).shape == (1, 27)


# -------------------------------------------------------- ResourceField
def test_resource_equilibrium():
    g = Grid((5, 5), L=10.0)
    rf = ResourceField(g, Gamma=0.001, gamma=0.1, epsilon=0.1, dt=0.025)
    assert rf.N_eq == pytest.approx(10.0)


def test_resource_field_relaxes_toward_equilibrium():
    rng = np.random.default_rng(0)
    g = Grid((6, 6), L=10.0)
    rf = ResourceField(g, Gamma=0.001, gamma=0.1, epsilon=0.1, dt=0.025)
    rf.seed(rng, n0=0)                  # start empty
    for _ in range(4000):
        rf.step(rng)
    mean_per_region = rf.total() / g.n_regions
    # equilibrium is 10; allow a generous stochastic band
    assert 6.0 < mean_per_region < 14.0


def test_harvest_decrements_count():
    rng = np.random.default_rng(1)
    g = Grid((3, 3), L=10.0)
    rf = ResourceField(g, Gamma=0.01, gamma=0.1, epsilon=0.1, dt=0.025)
    rf.seed(rng, n0=5)
    before = rf.total()
    disp, rids, slots = rf.neighbourhood(np.array([15.0, 15.0]))
    assert disp.shape[0] > 0
    assert rf.harvest(int(rids[0]), int(slots[0])) is True
    assert rf.total() == before - 1


# ------------------------------------------------------------ SpatialHash
def test_spatialhash_query_within():
    g = Grid((5, 5), L=10.0)
    pts = np.array([[1.0, 1.0], [12.0, 3.0], [1.5, 1.5]])
    sh = SpatialHash(g, pts)
    idx = sh.query_within(np.array([1.2, 1.2]), radius=2.0)
    assert sorted(idx.tolist()) == [0, 2]


# ------------------------------------------------------------- Population
def test_agent_starves_without_food():
    rng = np.random.default_rng(2)
    g = Grid((5, 5), L=10.0)
    pop = Population(g, v=20.0, dt=0.025, r_collect=1.0, R_sense=6.0,
                    mu0=1.0, s_max=1.0)
    pop.add([25.0, 25.0], heading=[1.0, 0.0], fuel=1.0)
    # no resource field -> pure metabolic decay, dies within s_max/(mu0*dt) steps
    for _ in range(100):
        pop.step([], rng)
        if pop.count == 0:
            break
    assert pop.count == 0


def test_ids_are_stable_through_death():
    rng = np.random.default_rng(3)
    g = Grid((5, 5), L=10.0)
    pop = Population(g, v=1.0, dt=0.025, r_collect=1.0, R_sense=6.0,
                    mu0=1.0, s_max=1.0)
    a = pop.add([10.0, 10.0], heading=[1.0, 0.0], fuel=0.01)
    b = pop.add([20.0, 20.0], heading=[1.0, 0.0], fuel=1.0)
    pop.step([], rng)                  # agent a (fuel < mu0*dt) dies this step
    assert pop.index_of(a) is None
    assert pop.index_of(b) is not None


# ------------------------------------------------------------- Simulation
def test_seeded_runs_are_reproducible():
    s1 = Simulation.forager(seed=42)
    s2 = Simulation.forager(seed=42)
    s1.run(300, stop_on_extinction=False)
    s2.run(300, stop_on_extinction=False)
    p1, p2 = s1.populations[0], s2.populations[0]
    assert p1.count == p2.count
    assert s1.fields[0].total() == s2.fields[0].total()
    if p1.count:
        assert np.allclose(p1.pos, p2.pos)


def test_forager_runs_in_2d_and_3d():
    s2 = Simulation.forager(seed=0, shape=(5, 5))
    s3 = Simulation.forager(seed=0, shape=(4, 4, 4))
    s2.run(100, stop_on_extinction=False)
    s3.run(100, stop_on_extinction=False)
    assert s2.grid.D == 2 and s3.grid.D == 3


def test_forager_population_is_sustainable():
    # With the applet parameters the greedy forager should boom-bust into a
    # sustained oscillation, not go extinct. Check it survives the transient.
    sim = Simulation.forager(seed=0)
    sim.run(900, stop_on_extinction=True)
    assert sim.populations[0].count > 0
    assert sim.step_count == 900           # never tripped the extinction stop


def test_homogeneity_sets_resource_density():
    # epsilon is derived from Xi so that N_eq per region = (Xi/r_col)^2 * L^2.
    s = Simulation.forager(seed=0, Xi=0.5)            # -> 0.25 * 100 = 25
    assert s.fields[0].N_eq == pytest.approx(25.0)
    s2 = Simulation.forager(seed=0, Xi=1.0)           # -> 1.0 * 100 = 100
    assert s2.fields[0].N_eq == pytest.approx(100.0)


def test_reproduction_increases_population():
    rng = np.random.default_rng(0)
    g = Grid((5, 5), L=10.0)
    pop = Population(g, v=20.0, dt=0.025, r_collect=1.0, R_sense=6.0,
                    mu0=1.0, s_max=1.0, repro_fraction=0.8)
    pop.add([25.0, 25.0], heading=[1.0, 0.0], fuel=1.0)  # above threshold
    n_before = pop.count
    pop._reproduce(rng)
    assert pop.count == n_before + 1
    assert pop.offspring[0] == 1


# ------------------------------------------------------------- Viewer
def _viewer(sim, **kw):
    """Build a viewer's figure on the Agg backend, without showing it."""
    v = Viewer(sim, **kw)
    v._build_figure()
    v.fig.canvas.draw()                # fixes the projection used by picking
    return v


def _world(shape, agents):
    """A small deterministic world with agents at the given positions."""
    rng = np.random.default_rng(0)
    g = Grid(shape, L=10.0)
    rf = ResourceField(g, Gamma=0.001, gamma=0.1, epsilon=0.04, dt=0.02)
    rf.seed(rng)
    pop = Population(g, v=20.0, dt=0.02, r_collect=1.0, R_sense=6.0,
                     mu0=0.1, s_max=1.0)
    for p in agents:
        pop.add(p, heading=np.eye(g.D)[0])
    return Simulation(g, [rf], [pop], dt=0.02, rng=rng)


def test_viewer_rejects_other_dimensions():
    fake = SimpleNamespace(grid=SimpleNamespace(D=4))
    with pytest.raises(ValueError, match="D = 4"):
        Viewer(fake)


@pytest.mark.parametrize("shape", [(5, 5), (4, 4, 4)])
def test_viewer_animates_frames(shape, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **kw: None)
    sim = Simulation.forager(seed=0, shape=shape)
    v = Viewer(sim)
    v.play()                          # builds the window and the animation
    assert isinstance(v._anim, FuncAnimation)

    v.selected_id = int(sim.populations[0].ids[0])
    before = sim.step_count
    for frame in range(3):
        v._update(frame)
        v.fig.canvas.draw()
    assert sim.step_count >= before + 3 * v.steps_per_frame
    assert "t = " in v._title.get_text()
    plt.close(v.fig)


def test_viewer_2d_click_selects_then_adds():
    sim = _world((5, 5), [[10.0, 10.0], [40.0, 40.0]])
    pop = sim.populations[0]
    v = _viewer(sim)

    v._on_click(SimpleNamespace(inaxes=v.ax, xdata=40.5, ydata=40.0))
    assert v.selected_id == 1

    v._on_click(SimpleNamespace(inaxes=v.ax, xdata=25.0, ydata=25.0))
    assert v.selected_id is None
    assert pop.count == 3
    assert np.allclose(pop.pos[2], [25.0, 25.0])
    plt.close(v.fig)


def test_viewer_3d_pick_selects_nearest_agent_within_tolerance():
    sim = _world((4, 4, 4), [[5.0, 5.0, 5.0], [35.0, 5.0, 35.0]])
    v = _viewer(sim)
    px, py = v._to_screen(sim.populations[0].pos[1:2])[0]

    v._pick(px, py)
    assert v.selected_id == 1

    # a pick just inside the screen tolerance still selects the same agent
    v.selected_id = None
    v._pick(px + 0.5 * v._screen_tolerance(), py)
    assert v.selected_id == 1
    plt.close(v.fig)


def test_viewer_3d_pick_in_empty_space_adds_agent_at_centre_depth():
    sim = _world((4, 4, 4), [[5.0, 5.0, 5.0], [35.0, 5.0, 35.0]])
    pop = sim.populations[0]
    v = _viewer(sim)
    centre = 0.5 * sim.grid.extent
    px, py = v._to_screen(centre.reshape(1, 3))[0]
    # the existing agents must be nowhere near that pixel for this to be a miss
    assert np.linalg.norm(v._to_screen(pop.pos) - [px, py],
                          axis=1).min() > v._screen_tolerance()

    v._pick(px, py)
    assert pop.count == 3
    assert np.allclose(pop.pos[2], centre)
    assert v.selected_id is None

    # off-centre pixels land on the same depth plane, still inside the box
    px2, py2 = px + 40.0, py - 25.0
    v._pick(px2, py2)
    assert pop.count == 4
    added = pop.pos[3]
    assert np.all(added >= 0.0) and np.all(added <= sim.grid.extent)
    assert np.allclose(v._to_screen(added.reshape(1, 3))[0], [px2, py2])
    plt.close(v.fig)


def test_viewer_3d_rotating_drag_is_not_a_pick():
    sim = _world((4, 4, 4), [[5.0, 5.0, 5.0], [35.0, 5.0, 35.0]])
    pop = sim.populations[0]
    v = _viewer(sim)
    px, py = v._to_screen(pop.pos[1:2])[0]

    v._on_press(SimpleNamespace(inaxes=v.ax, button=1, x=px, y=py))
    v._on_release(SimpleNamespace(inaxes=v.ax, button=1, x=px + 60, y=py + 40))
    assert v.selected_id is None and pop.count == 2

    v._on_press(SimpleNamespace(inaxes=v.ax, button=1, x=px, y=py))
    v._on_release(SimpleNamespace(inaxes=v.ax, button=1, x=px + 1, y=py))
    assert v.selected_id == 1
    plt.close(v.fig)


def test_viewer_3d_spacebar_pauses():
    sim = Simulation.forager(seed=0, shape=(4, 4, 4))
    v = _viewer(sim)
    v._on_key(SimpleNamespace(key=" "))
    assert v.paused
    v._update(0)
    assert sim.step_count == 0
    v._on_key(SimpleNamespace(key=" "))
    v._update(1)
    assert sim.step_count == v.steps_per_frame
    plt.close(v.fig)
