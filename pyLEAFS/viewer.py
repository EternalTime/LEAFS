"""Interactive matplotlib viewer for a 2d or 3d simulation.

Controls
--------
spacebar
    Pause / resume the simulation.
click empty space
    Add a new agent at the cursor (works while paused or running).
click on an agent
    Select it for inspection: a highlight ring appears, a side panel shows its
    state, and its recent trajectory is drawn as a trail. Click empty space
    again to deselect (or to add an agent there).

In 3d the same controls apply, with left-drag reserved for rotating the view;
see :class:`Viewer` for how a click is told apart from a rotation.

The viewer reads the simulation through its public attributes only; it never
reaches into private buffers. The grid's ``D`` selects the axes it builds.
"""

import numpy as np

from pyLEAFS import palette

# A press and release within this many pixels of each other is a click, not a
# drag; anything larger in 3d is the axes' own rotation gesture.
_CLICK_SLOP_PX = 3.0

# In 3d the resource field is split into this many equal-count depth bands,
# each drawn as its own translucent layer, so the volume reads as haze with
# distance rather than as an opaque wall of near points.
_HAZE_BANDS = 8
_HAZE_ALPHA = (0.04, 0.32)      # farthest band, nearest band
_HAZE_SIZE = (0.5, 1.1)         # marker size in points, same order

# Layout, in figure fractions: the side panel is only as wide as its monospace
# text and the scene fills everything else.
_PANEL_WIDTH = 0.318
_SCENE_RECT = (0.023, 0.021, 0.640, 0.879)

# The widest and tallest the world box's projection ever gets, as a fraction of
# the 3d axes it lives in, taken over every elevation and azimuth: a box seen
# down a body diagonal covers far more of the axes than one seen face-on, so
# the 3d axes is inflated by the reciprocal of the *worst* case. Fitting the
# default view instead would leave the box clipped as soon as it was dragged.
#
# Both numbers are closed-form maxima rather than the largest value on a
# sampled sweep, which straddles the true extreme without ever landing on it.
# mplot3d normalises any world to a box of aspect 4:4:3 and looks at it from a
# fixed distance, so the widest view is down the z axis with the near face
# turned corner-on, and the tallest looks along the box's own body diagonal,
# whose direction sets elev = atan(4 * sqrt(2) / 3). Both worst cases are
# independent of the world's own shape, because of that normalisation.
_WORST_VIEW_3D = ((90.0, 45.0), (62.0616, 45.0))        # widest, tallest
_SCENE_FILL_3D = (0.9526, 1.0302)


def _inflate(rect, fx, fy):
    """Grow a ``(x, y, w, h)`` figure rectangle about its centre."""
    x, y, w, h = rect
    return (x - 0.5 * (fx - 1.0) * w, y - 0.5 * (fy - 1.0) * h, w * fx, h * fy)


def _unclipped(artist):
    """Let a 3d artist draw over the whole scene rectangle.

    Matplotlib squares off a 3d axes and clips its artists to that square,
    which is shorter than the box's own projection at a steep elevation; the
    tip of the world would be sliced off mid-drag. ``_SCENE_FILL_3D`` is what
    keeps the box inside the scene rectangle instead.
    """
    artist.set_clip_on(False)
    return artist


class Viewer:
    """An interactive viewer for a :class:`~pyLEAFS.simulation.Simulation`.

    ``D == 2`` draws a flat box, ``D == 3`` a rotatable one; both offer the
    same controls, side panel, and palette.

    Parameters
    ----------
    sim : Simulation
        The simulation to display. Must be 2- or 3-dimensional.
    steps_per_frame : int, optional
        Simulation steps advanced per rendered frame (default 4).
    select_radius : float, optional
        Click-to-select tolerance in physical units; a click within this
        distance of an agent selects it, otherwise it adds a new agent
        (default 2.0). In 3d the tolerance is the same physical length,
        projected to pixels at the centre of the box.
    trail_length : int, optional
        Number of past positions retained for the selected agent's trail
        (default 200).
    interval : int, optional
        Matplotlib animation interval in milliseconds (default 30).

    Examples
    --------
    >>> from pyLEAFS import Simulation, Viewer                            # doctest: +SKIP
    >>> Viewer(Simulation.forager(seed=0)).play()                         # doctest: +SKIP
    >>> Viewer(Simulation.forager(seed=0, shape=(10, 10, 10))).play()     # doctest: +SKIP

    Controls (3d)
    -------------
    left-drag
        Rotate the view, as any 3d axes does. Rotating never selects or adds.
    click empty space
        A press and release at effectively the same place is a pick. A pick
        that lands near no agent adds one on the line of sight, at the depth
        of the centre of the world box.
    click on an agent
        A pick selects the agent nearest the line of sight, provided it is
        within ``select_radius`` of it as measured on screen.
    """

    def __init__(self, sim, steps_per_frame=4, select_radius=2.0,
                 trail_length=200, interval=30):
        self.D = sim.grid.D
        if self.D not in (2, 3):
            raise ValueError(
                "Viewer supports 2d or 3d simulations only, got D = %d" % self.D
            )
        self.sim = sim
        self.steps_per_frame = int(steps_per_frame)
        self.select_radius = float(select_radius)
        self.trail_length = int(trail_length)
        self.interval = int(interval)

        self.paused = False
        self.selected_id = None
        self._trail = []
        self._press_px = None

    # --------------------------------------------------------- main loop
    def play(self):
        """Open the window and run until it is closed."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        self._build_figure()
        self._anim = FuncAnimation(self.fig, self._update,
                                   interval=self.interval, blit=False,
                                   cache_frame_data=False)
        plt.show()

    def _build_figure(self):
        """Create the figure, the axes, the artists, and the event hooks."""
        import matplotlib.pyplot as plt
        # registers the '3d' projection without relying on a lazy import
        import mpl_toolkits.mplot3d                                # noqa: F401

        self.fig = plt.figure(figsize=(8.9, 7.0))
        self.fig.patch.set_facecolor(palette.background)
        self.fig.canvas.manager.set_window_title("pyLEAFS")

        x, y, w, h = _SCENE_RECT
        if self.D == 2:
            self.ax = self.fig.add_axes(_SCENE_RECT)
            self._build_axes_2d()
        else:
            self.ax = self.fig.add_axes(
                _inflate(_SCENE_RECT, 1.0 / _SCENE_FILL_3D[0],
                         1.0 / _SCENE_FILL_3D[1]),
                projection="3d")
            self._build_axes_3d()
        self._title = self.fig.text(x + 0.5 * w, y + h + 0.038, "",
                                    ha="center", va="center", fontsize=11,
                                    color=palette.sensor_mist)

        self.panel = self.fig.add_axes([1.0 - _PANEL_WIDTH, 0.0,
                                        _PANEL_WIDTH, 1.0])
        self.panel.set_facecolor(palette.background)
        self.panel.axis("off")
        self._panel_text = self.panel.text(
            0.0, y + h, "", va="top", ha="left", family="monospace",
            fontsize=8, color=palette.sensor_mist,
            bbox=dict(boxstyle="round,pad=0.6", facecolor=palette.abyss_blue,
                      edgecolor=palette.sensor_slate, linewidth=0.8),
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        if self.D == 2:
            self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        else:
            self.fig.canvas.mpl_connect("button_press_event", self._on_press)
            self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    def _build_axes_2d(self):
        ext = self.sim.grid.extent
        self.ax.set_facecolor(palette.background)
        self.ax.set_xlim(0, ext[0])
        self.ax.set_ylim(0, ext[1])
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color(palette.sensor_slate)

        # artists, created empty and updated in place
        self._resource_scatter = self.ax.scatter([], [], s=18, marker="*",
                                                  color=palette.food_green)
        self._agent_scatter = self.ax.scatter([], [], s=40,
                                               color=palette.motor_wine)
        self._trail_line, = self.ax.plot([], [], "-", lw=1.0,
                                         color=palette.brain_pink, alpha=0.9)
        self._ring = self.ax.scatter([], [], s=200, facecolors="none",
                                     edgecolors=palette.sensor_mist, lw=2.0)

    def _build_axes_3d(self):
        ext = self.sim.grid.extent
        self.ax.set_facecolor(palette.background)
        self.ax.set_xlim(0, ext[0])
        self.ax.set_ylim(0, ext[1])
        self.ax.set_zlim(0, ext[2])
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color(palette.background + (1.0,))
            axis.line.set_color(palette.sensor_slate)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

        # Marker-only Line3D layers draw the whole resource field an order of
        # magnitude faster than a 3d scatter, which depth-sorts every point,
        # and one layer per depth band is what makes the volume see-through.
        self._resource_bands = []
        for k in range(_HAZE_BANDS):
            near = k / (_HAZE_BANDS - 1)      # 0 = farthest band, 1 = nearest
            band, = self.ax.plot(
                [], [], [], linestyle="none", marker="o", markeredgewidth=0,
                ms=_HAZE_SIZE[0] + near * (_HAZE_SIZE[1] - _HAZE_SIZE[0]),
                alpha=_HAZE_ALPHA[0] + near * (_HAZE_ALPHA[1] - _HAZE_ALPHA[0]),
                color=palette.food_green,
            )
            self._resource_bands.append(_unclipped(band))
        self._trail_line, = self.ax.plot([], [], [], "-", lw=1.4,
                                         color=palette.brain_pink, alpha=0.9)
        _unclipped(self._trail_line)
        # agents and the highlight ring are rebuilt per frame; see _draw_3d
        self._agent_scatter = None
        self._ring = None

    # ------------------------------------------------------------ events
    def _on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        click = np.array([event.xdata, event.ydata])
        pop = self.sim.populations[0]
        if pop.count > 0:
            disp = self.sim.grid.displacement(click.reshape(1, 2), pop.pos)
            d2 = np.einsum("ij,ij->i", disp, disp)
            j = int(np.argmin(d2))
            if d2[j] <= self.select_radius ** 2:
                self.selected_id = int(pop.ids[j])
                self._trail = []
                return
        # empty space: add an agent there, and deselect
        pop.add(click, rng=self.sim.rng)
        self.selected_id = None
        self._trail = []

    def _on_press(self, event):
        self._press_px = None
        if event.inaxes is self.ax and event.button == 1:
            self._press_px = (event.x, event.y)

    def _on_release(self, event):
        press, self._press_px = self._press_px, None
        if press is None or event.inaxes is not self.ax:
            return
        moved = max(abs(event.x - press[0]), abs(event.y - press[1]))
        if moved > _CLICK_SLOP_PX:
            return                      # the view was rotated, not clicked
        self._pick(event.x, event.y)

    def _pick(self, px, py):
        """Select the agent nearest the line of sight, else add one on it."""
        pop = self.sim.populations[0]
        if pop.count > 0:
            screen = self._to_screen(pop.pos)
            d = screen - np.array([px, py])
            d2 = np.einsum("ij,ij->i", d, d)
            j = int(np.argmin(d2))
            if d2[j] <= self._screen_tolerance() ** 2:
                self.selected_id = int(pop.ids[j])
                self._trail = []
                return
        pop.add(self._ray_point(px, py), rng=self.sim.rng)
        self.selected_id = None
        self._trail = []

    # ------------------------------------------------- 3d projection maths
    def _to_screen(self, pos):
        """Project world positions ``(n, 3)`` to pixel coordinates ``(n, 2)``."""
        from mpl_toolkits.mplot3d import proj3d

        pos = np.atleast_2d(pos)
        x, y, _ = proj3d.proj_transform(pos[:, 0], pos[:, 1], pos[:, 2],
                                        self.ax.get_proj())
        return self.ax.transData.transform(np.column_stack([x, y]))

    def _depths(self, pos):
        """Projected depth of world positions ``(n, 3)``; smaller is nearer."""
        M = self.ax.get_proj()
        v = np.column_stack([pos, np.ones(len(pos))])
        return (v @ M[2]) / (v @ M[3])

    def _screen_tolerance(self):
        """``select_radius`` in pixels, measured at the centre of the box."""
        centre = 0.5 * self.sim.grid.extent
        origin = self._to_screen(centre)[0]
        steps = centre + self.select_radius * np.eye(3)
        return float(np.linalg.norm(self._to_screen(steps) - origin,
                                    axis=1).max())

    def _ray_point(self, px, py):
        """The point under pixel ``(px, py)`` at the depth of the box centre."""
        from mpl_toolkits.mplot3d import proj3d

        M = self.ax.get_proj()
        centre = 0.5 * self.sim.grid.extent
        _, _, depth = proj3d.proj_transform(centre[0], centre[1], centre[2], M)
        xd, yd = self.ax.transData.inverted().transform((px, py))
        world = np.linalg.solve(M, np.array([xd, yd, depth, 1.0]))
        return world[:3] / world[3]

    # ------------------------------------------------------------ render
    def _update(self, _frame):
        if not self.paused:
            for _ in range(self.steps_per_frame):
                self.sim.step()

        pop = self.sim.populations[0]
        resource = self.sim.fields[0]

        if self.D == 2:
            self._draw_2d(pop, resource)
        else:
            self._draw_3d(pop, resource)
        self._render_panel(pop)

        status = "PAUSED" if self.paused else "running"
        extinct = " — EXTINCTION" if pop.count == 0 else ""
        self._title.set_text(
            f"t = {self.sim.time:7.2f}   agents = {pop.count}   "
            f"resources = {resource.total()}   [{status}]{extinct}"
        )
        return ()

    def _draw_2d(self, pop, resource):
        self._resource_scatter.set_offsets(resource.all_positions())

        if pop.count:
            self._agent_scatter.set_offsets(pop.pos.copy())
            # marker size scales with fuel fraction
            frac = np.clip(pop.fuel / pop.s_max, 0.1, 1.0)
            self._agent_scatter.set_sizes(20 + 60 * frac)
        else:
            self._agent_scatter.set_offsets(np.empty((0, 2)))

        trail, selected = self._selection_trail(pop)
        if selected is None:
            self._ring.set_offsets(np.empty((0, 2)))
        else:
            self._ring.set_offsets(selected.reshape(1, 2))
        self._trail_line.set_data(trail[:, 0], trail[:, 1])

    def _draw_3d(self, pop, resource):
        self._draw_haze(resource.all_positions())

        # a 3d scatter has no in-place offset setter, so rebuild the few
        # agent-sized artists each frame
        for artist in (self._agent_scatter, self._ring):
            if artist is not None:
                artist.remove()
        self._agent_scatter = self._ring = None

        if pop.count:
            frac = np.clip(pop.fuel / pop.s_max, 0.1, 1.0)
            self._agent_scatter = _unclipped(self.ax.scatter(
                pop.pos[:, 0], pop.pos[:, 1], pop.pos[:, 2],
                s=14 + 30 * frac, color=palette.motor_wine,
                edgecolors=palette.sensor_mist, linewidths=0.6,
                depthshade=False,
            ))

        trail, selected = self._selection_trail(pop)
        if selected is not None:
            self._ring = _unclipped(self.ax.scatter(
                [selected[0]], [selected[1]], [selected[2]], s=200,
                facecolors="none", edgecolors=palette.sensor_mist, lw=2.0,
                depthshade=False,
            ))
        self._trail_line.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])

    def _draw_haze(self, rpos):
        """Split the resource points into equal-count bands, near band last.

        Each band keeps a fixed alpha and marker size, so sorting the points
        into bands by their projected depth is what fades the far side of the
        volume and leaves the near side brightest.
        """
        if rpos.shape[0] < _HAZE_BANDS:
            for band in self._resource_bands:
                band.set_data_3d(rpos[:0, 0], rpos[:0, 1], rpos[:0, 2])
            self._resource_bands[-1].set_data_3d(rpos[:, 0], rpos[:, 1],
                                                 rpos[:, 2])
            return
        depth = self._depths(rpos)
        edges = np.quantile(depth, np.linspace(0.0, 1.0, _HAZE_BANDS + 1))
        edges[0] -= 1.0
        edges[-1] += 1.0
        # smaller depth is nearer the viewer, so the last band is the nearest
        rank = np.searchsorted(edges, depth, side="right") - 1
        rank = _HAZE_BANDS - 1 - np.clip(rank, 0, _HAZE_BANDS - 1)
        for k, band in enumerate(self._resource_bands):
            sel = rpos[rank == k]
            band.set_data_3d(sel[:, 0], sel[:, 1], sel[:, 2])

    def _selection_trail(self, pop):
        """Advance the selected agent's trail; return ``(trail, position)``.

        ``trail`` is an ``(n, D)`` array of past positions, broken by NaNs
        wherever the agent wrapped across a toroidal boundary, and
        ``position`` is the agent's current position, or None if nothing is
        selected (the trail is then empty).
        """
        empty = np.empty((0, self.D))
        if self.selected_id is None:
            return empty, None
        idx = pop.index_of(self.selected_id)
        if idx is None:                 # selected agent died
            self.selected_id = None
            self._trail = []
            return empty, None

        p = pop.pos[idx].copy()
        self._trail.append(p)
        if len(self._trail) > self.trail_length:
            self._trail = self._trail[-self.trail_length:]
        trail = np.array(self._trail, dtype=float)
        if len(trail) > 1:
            jumps = np.linalg.norm(np.diff(trail, axis=0), axis=1)
            cut = jumps > 0.5 * self.sim.grid.extent.min()
            trail[1:][cut] = np.nan
        return trail, p

    def _render_panel(self, pop):
        if self.selected_id is None:
            self._panel_text.set_text(
                "No agent selected.\n\n"
                "spacebar : pause/resume\n"
                "click agent : inspect\n"
                "click empty : add agent"
            )
            return
        idx = pop.index_of(self.selected_id)
        if idx is None:
            self._panel_text.set_text("(selected agent died)")
            return
        p = pop.pos[idx]
        h = pop.heading[idx]
        if self.D == 2:
            where = f"({p[0]:6.2f}, {p[1]:6.2f})"
            heading = f"{np.degrees(np.arctan2(h[1], h[0])):6.1f} deg"
        else:
            where = f"({p[0]:6.2f}, {p[1]:6.2f}, {p[2]:6.2f})"
            azimuth = np.degrees(np.arctan2(h[1], h[0]))
            elevation = np.degrees(np.arcsin(np.clip(h[2], -1.0, 1.0)))
            heading = f"{azimuth:6.1f} az {elevation:6.1f} el"
        self._panel_text.set_text(
            f"agent id   : {self.selected_id}\n"
            f"position   : {where}\n"
            f"heading    : {heading}\n"
            f"fuel       : {pop.fuel[idx]:6.3f} / {pop.s_max:.2f}\n"
            f"age        : {pop.age[idx]} steps\n"
            f"harvested  : {pop.harvested[idx]}\n"
            f"offspring  : {pop.offspring[idx]}"
        )
