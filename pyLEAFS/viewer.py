"""Interactive matplotlib viewer for a 2d simulation.

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

The viewer reads the simulation through its public attributes only; it never
reaches into private buffers. It targets ``D == 2``; a 3d viewer is a later
layer.
"""

import numpy as np

from pyLEAFS import palette


class Viewer:
    """An interactive viewer for a 2d :class:`~pyLEAFS.simulation.Simulation`.

    Parameters
    ----------
    sim : Simulation
        The simulation to display. Must be 2-dimensional.
    steps_per_frame : int, optional
        Simulation steps advanced per rendered frame (default 4).
    select_radius : float, optional
        Click-to-select tolerance in physical units; a click within this
        distance of an agent selects it, otherwise it adds a new agent
        (default 2.0).
    trail_length : int, optional
        Number of past positions retained for the selected agent's trail
        (default 200).
    interval : int, optional
        Matplotlib animation interval in milliseconds (default 30).

    Examples
    --------
    >>> from pyLEAFS import Simulation, Viewer       # doctest: +SKIP
    >>> Viewer(Simulation.forager(seed=0)).play()    # doctest: +SKIP
    """

    def __init__(self, sim, steps_per_frame=4, select_radius=2.0,
                 trail_length=200, interval=30):
        if sim.grid.D != 2:
            raise ValueError("Viewer supports 2d simulations only")
        self.sim = sim
        self.steps_per_frame = int(steps_per_frame)
        self.select_radius = float(select_radius)
        self.trail_length = int(trail_length)
        self.interval = int(interval)

        self.paused = False
        self.selected_id = None
        self._trail = []

    # --------------------------------------------------------- main loop
    def play(self):
        """Open the window and run until it is closed."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        ext = self.sim.grid.extent
        self.fig, (self.ax, self.panel) = plt.subplots(
            1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [3, 1]}
        )
        self.fig.canvas.manager.set_window_title("pyLEAFS")

        self.ax.set_facecolor(palette.background)
        self.ax.set_xlim(0, ext[0])
        self.ax.set_ylim(0, ext[1])
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # artists, created empty and updated in place
        self._resource_scatter = self.ax.scatter([], [], s=18, marker="*",
                                                  color=palette.food_green)
        self._agent_scatter = self.ax.scatter([], [], s=40,
                                               color=palette.motor_wine)
        self._trail_line, = self.ax.plot([], [], "-", lw=1.0,
                                         color=palette.brain_pink, alpha=0.9)
        self._ring = self.ax.scatter([], [], s=200, facecolors="none",
                                     edgecolors=palette.sensor_mist, lw=2.0)
        self._title = self.ax.set_title("", color="white")

        self.panel.axis("off")
        self._panel_text = self.panel.text(
            0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=9
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self._anim = FuncAnimation(self.fig, self._update,
                                   interval=self.interval, blit=False,
                                   cache_frame_data=False)
        plt.show()

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

    # ------------------------------------------------------------ render
    def _update(self, _frame):
        if not self.paused:
            for _ in range(self.steps_per_frame):
                self.sim.step()

        pop = self.sim.populations[0]
        resource = self.sim.fields[0]

        rpos = resource.all_positions()
        self._resource_scatter.set_offsets(rpos if rpos.size else np.empty((0, 2)))

        if pop.count:
            self._agent_scatter.set_offsets(pop.pos.copy())
            # marker size scales with fuel fraction
            frac = np.clip(pop.fuel / pop.s_max, 0.1, 1.0)
            self._agent_scatter.set_sizes(20 + 60 * frac)
        else:
            self._agent_scatter.set_offsets(np.empty((0, 2)))

        self._render_selection(pop)
        self._render_panel(pop)

        status = "PAUSED" if self.paused else "running"
        extinct = " — EXTINCTION" if pop.count == 0 else ""
        self._title.set_text(
            f"t = {self.sim.time:7.2f}   agents = {pop.count}   "
            f"resources = {resource.total()}   [{status}]{extinct}"
        )
        return ()

    def _render_selection(self, pop):
        if self.selected_id is None:
            self._ring.set_offsets(np.empty((0, 2)))
            self._trail_line.set_data([], [])
            return
        idx = pop.index_of(self.selected_id)
        if idx is None:
            # selected agent died
            self.selected_id = None
            self._ring.set_offsets(np.empty((0, 2)))
            self._trail_line.set_data([], [])
            self._trail = []
            return
        p = pop.pos[idx].copy()
        self._ring.set_offsets(p.reshape(1, 2))
        # trail: break the line across toroidal wraps to avoid long jumps
        self._trail.append(p)
        if len(self._trail) > self.trail_length:
            self._trail = self._trail[-self.trail_length:]
        trail = np.array(self._trail)
        if len(trail) > 1:
            jumps = np.linalg.norm(np.diff(trail, axis=0), axis=1)
            cut = jumps > 0.5 * self.sim.grid.extent.min()
            tx, ty = trail[:, 0].astype(float), trail[:, 1].astype(float)
            tx = tx.copy(); ty = ty.copy()
            tx[1:][cut] = np.nan
            ty[1:][cut] = np.nan
            self._trail_line.set_data(tx, ty)

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
        heading_deg = np.degrees(np.arctan2(h[1], h[0]))
        self._panel_text.set_text(
            f"agent id   : {self.selected_id}\n"
            f"position   : ({p[0]:6.2f}, {p[1]:6.2f})\n"
            f"heading    : {heading_deg:6.1f} deg\n"
            f"fuel       : {pop.fuel[idx]:6.3f} / {pop.s_max:.2f}\n"
            f"age        : {pop.age[idx]} steps\n"
            f"harvested  : {pop.harvested[idx]}\n"
            f"offspring  : {pop.offspring[idx]}"
        )
