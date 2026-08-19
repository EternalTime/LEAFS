"""Tests pinning the physics and bookkeeping of pyLEAFS v1."""

import itertools
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.animation import FuncAnimation                      # noqa: E402

from pyLEAFS import (Grid, ResourceField, SpatialHash, Population,  # noqa: E402
                     Simulation, Viewer, palette)
from pyLEAFS.viewer import _scene_fill_3d, _WORST_VIEW_3D       # noqa: E402


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
    # epsilon is derived from Xi so that N_eq per region = (Xi/r_col)^D * L^D.
    s = Simulation.forager(seed=0, Xi=0.5)            # -> 0.25 * 100 = 25
    assert s.fields[0].N_eq == pytest.approx(25.0)
    s2 = Simulation.forager(seed=0, Xi=1.0)           # -> 1.0 * 100 = 100
    assert s2.fields[0].N_eq == pytest.approx(100.0)


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 10)])
@pytest.mark.parametrize("Xi", [0.3, 0.5, 1.0])
def test_homogeneity_exponent_follows_the_dimension(shape, Xi):
    # Theory: Xi is r_col over the mean resource spacing, so the equilibrium
    # number density is (Xi/r_col)^D in any dimension, not just 2d.
    sim = Simulation.forager(seed=0, shape=shape, Xi=Xi)
    field, D, r_collect = sim.fields[0], sim.grid.D, 1.0
    density = field.Gamma / (field.epsilon * field.gamma)
    assert density == pytest.approx((Xi / r_collect) ** D)
    assert field.N_eq == pytest.approx(density * sim.grid.L ** D)
    spacing = density ** (-1.0 / D)
    assert spacing == pytest.approx(r_collect / Xi)


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


def _scene_corners(v):
    """Screen positions of the corners of the drawn world box."""
    corners = np.array(list(itertools.product(
        *[(0.0, e) for e in v.sim.grid.extent])))
    if v.D == 3:
        return v._to_screen(corners)
    return v.ax.transData.transform(corners)


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


@pytest.mark.parametrize("shape", [(5, 5), (4, 4, 4)])
def test_viewer_chrome_uses_the_palette(shape):
    sim = Simulation.forager(seed=0, shape=shape)
    v = _viewer(sim)
    assert v.fig.get_facecolor()[:3] == pytest.approx(palette.background)
    assert v._title.get_color() == palette.sensor_mist
    assert v._panel_text.get_color() == palette.sensor_mist
    assert v._panel_text.get_bbox_patch() is not None      # the panel card
    if v.D == 3:
        for axis in (v.ax.xaxis, v.ax.yaxis, v.ax.zaxis):
            assert axis.line.get_color() == palette.sensor_slate
    else:
        for spine in v.ax.spines.values():
            assert spine.get_edgecolor()[:3] == pytest.approx(palette.sensor_slate)
    plt.close(v.fig)


def test_viewer_3d_haze_bands_split_resources_by_depth():
    sim = Simulation.forager(seed=0, shape=(4, 4, 4))
    v = _viewer(sim)
    v._update(0)

    counts, mean_depth = [], []
    for band in v._resource_bands:
        pts = np.column_stack(band.get_data_3d())
        counts.append(len(pts))
        mean_depth.append(v._depths(pts).mean())
    assert sum(counts) == sim.fields[0].total()            # every point drawn
    assert min(counts) > 0
    # bands run far to near, and nearer points are drawn brighter and larger
    assert mean_depth == sorted(mean_depth, reverse=True)
    alphas = [band.get_alpha() for band in v._resource_bands]
    sizes = [band.get_markersize() for band in v._resource_bands]
    assert alphas == sorted(alphas) and alphas[0] < alphas[-1]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]
    plt.close(v.fig)


def test_viewer_3d_haze_survives_a_nearly_empty_field():
    sim = Simulation.forager(seed=0, shape=(4, 4, 4))
    v = _viewer(sim)

    v._draw_haze(np.empty((0, 3)))
    assert all(len(b.get_data_3d()[0]) == 0 for b in v._resource_bands)

    sparse = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v._draw_haze(sparse)
    drawn = np.column_stack(v._resource_bands[-1].get_data_3d())
    assert np.allclose(drawn, sparse)
    assert sum(len(b.get_data_3d()[0]) for b in v._resource_bands) == 2
    v.fig.canvas.draw()
    plt.close(v.fig)


@pytest.mark.parametrize("shape", [(5, 5), (4, 4, 4)])
def test_viewer_scene_fills_the_window(shape):
    sim = Simulation.forager(seed=0, shape=shape)
    v = _viewer(sim)
    v.selected_id = int(sim.populations[0].ids[0])
    v._update(0)
    v.fig.canvas.draw()
    w, h = v.fig.canvas.get_width_height()

    drawn = _scene_corners(v)
    x0, y0 = drawn.min(axis=0)
    x1, y1 = drawn.max(axis=0)
    assert x1 - x0 > 0.55 * w and y1 - y0 > 0.70 * h

    card = v._panel_text.get_bbox_patch().get_window_extent(
        v.fig.canvas.get_renderer())
    assert card.x0 >= x1                       # the panel clears the scene
    assert card.x1 <= w and card.y1 <= h       # and stays inside the window
    assert card.y1 > 0.8 * h                   # anchored at the top of it
    title_y = v._title.get_window_extent(v.fig.canvas.get_renderer()).y0
    assert title_y >= y1
    plt.close(v.fig)


def _view_angles():
    """A sweep of view angles, plus the worst cases the sweep straddles."""
    angles = [(float(elev), float(azim))
              for elev in range(-90, 91, 15)
              for azim in range(-180, 181, 15)]
    return angles + [(se * elev, sa * azim + turn)
                     for elev, azim in _WORST_VIEW_3D
                     for se in (1.0, -1.0)
                     for sa in (1.0, -1.0)
                     for turn in (0.0, 180.0)]


def test_viewer_3d_box_stays_in_the_window_at_every_view_angle():
    # the 3d axes is inflated to fill the window, and a cube's projection grows
    # by up to about a factor of sqrt(2) as it turns, so check the extremes
    sim = Simulation.forager(seed=0, shape=(4, 4, 4))
    v = _viewer(sim)
    v._update(0)
    v.fig.canvas.draw()
    w, h = v.fig.canvas.get_width_height()
    card = v._panel_text.get_bbox_patch().get_window_extent(
        v.fig.canvas.get_renderer())

    for elev, azim in _view_angles():
        v.ax.view_init(elev, azim)
        drawn = _scene_corners(v)
        x0, y0 = drawn.min(axis=0)
        x1, y1 = drawn.max(axis=0)
        where = "elev=%g azim=%g" % (elev, azim)
        assert x0 >= 0 and y0 >= 0, where
        assert x1 <= w and y1 <= h, where
        assert x1 <= card.x0, where
    plt.close(v.fig)


def test_viewer_3d_fill_is_the_true_worst_case_over_all_view_angles():
    # the measured fill must be the largest the box ever gets, or the window
    # is sized for a view the user can rotate straight past
    sim = Simulation.forager(seed=0, shape=(4, 4, 4))
    v = _viewer(sim)
    v.fig.canvas.draw()
    box = v.ax.bbox

    fill = np.array([0.0, 0.0])
    for elev, azim in _view_angles():
        v.ax.view_init(elev, azim)
        drawn = _scene_corners(v)
        span = drawn.max(axis=0) - drawn.min(axis=0)
        fill = np.maximum(fill, span / [box.width, box.height])
    # the sweep repeats the worst-case projection, so it lands on the measured
    # value to within pixel-transform rounding rather than under it
    worst = np.array(_scene_fill_3d())
    assert np.all(fill <= worst + 1e-9)         # nothing overflows
    assert np.allclose(fill, worst, atol=5e-4)  # and nothing is lost
    plt.close(v.fig)


def test_viewer_3d_draws_the_tip_of_the_box_at_the_worst_view_angle():
    # matplotlib squares off a 3d axes, so at a steep elevation the box's own
    # projection is taller than the axes: an agent in the highest corner must
    # still be drawn, not sliced off at the edge of that square
    sim = Simulation.forager(seed=0, shape=(4, 4, 4))
    v = _viewer(sim)
    v.paused = True
    v.ax.view_init(*_WORST_VIEW_3D[1])

    ext = sim.grid.extent
    tip = _scene_corners(v)[:, 1].argmax()
    corner = np.array(list(itertools.product(*[(0.0, e) for e in ext])))[tip]
    corner = np.minimum(corner, ext - 1e-6)     # the far face wraps to zero
    sim.populations[0].add(corner, heading=np.eye(3)[0], rng=sim.rng)
    v._update(0)
    v.fig.canvas.draw()

    px, py = v._to_screen(corner.reshape(1, 3))[0]
    assert py > v.ax.bbox.y1               # the corner really is off the axes
    buf = np.asarray(v.fig.canvas.buffer_rgba())
    row = buf.shape[0] - 1 - int(round(py))
    patch = buf[row - 6:row + 7, int(round(px)) - 6:int(round(px)) + 7, :3]
    off = np.abs(patch / 255.0 - palette.motor_wine).max(axis=2)
    assert (off < 0.06).sum() > 10         # the agent's marker is really there
    plt.close(v.fig)
