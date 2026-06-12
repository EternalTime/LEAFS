# pyLEAFS — Port Summary

Python port of LEAFS (Layered Environment with Agents Foraging Simulator) from
MATLAB, structured as a git repository with Sphinx documentation, mirroring the
`EternalTime/CellularAutomata` template (flat package dir, `tests/`, `docs/`,
`matlab/` archive, `pyproject.toml`, MIT, README).

## Sources reviewed

### MATLAB codebase

Project root: `~/Documents/Research/LEAFS/Information Modulation in Sensor Dev/`

- `Codebase/` — primary source. `AgentRNN.m`, `Environment.m`, `Region.m`,
  `PopulationRNN.m`, `palette.m`, `run_replicate.m`, plus `Visualize.m`
  (multi-panel population-stats viewer to port) and `plot_genome_graph.m`
  (genome-graph rendering).
- `Sensors/Codebase/` — cluster deployment variant. `Environment.m`,
  `PopulationRNN.m`, `Region.m`, `palette.m` are **byte-identical** to the root
  `Codebase/` versions; only `AgentRNN.m` and `run_replicate.m` differ. Merge
  these two when populating the `matlab/` archive.
- `Sensors/leafs_array.slurm`, `submit.sh` — SLURM sweep infrastructure to port.
- `Test/TestCodebase.m` — interactive real-time visualiser reference.
- `codebase_summary.txt` (root) — prior architecture write-up.

### Web applets (use as porting references / validation targets)

Folder: `~/Documents/MyWebPage/assets/js/applets/`

- `forager_applet.js` — 2D greedy forager; Ξ homogeneity parameter; phase
  plane. **Primary reference for v1.**
- `forager3d_applet.js` — 3D generalisation; cleanest Ξ derivation (the header
  comment block is the clearest statement of the resource-field math).
- `leafs_applet.js` — full LEAFS: RNN controller + genome editor + region grid.
  Reference for the later neuroevolution and heterogeneous-environment layers.

(Self-contained applet HTML also exists under the project root: `Applet/`,
`leafs_applet.html`, `applet_params.txt`.)

### Research Statement

`~/...uploads/...Research_Statement.pdf` — LEAFS layer structure (resource layer
/ thermodynamic + pheromone layers / agent layer); sensor-development paper;
neuroevolution extension.

## Confirmed design decisions

- **Template:** mirror CellularAutomata repo layout.
- **Dimension-agnostic core:** D = 2 or 3 via parameter; positions as `(n, D)`
  arrays. Not separate 2D/3D codebases.
- **Stack:** NumPy, struct-of-arrays layout, Numba-ready but no `@njit` yet
  (flip on later by decorating hot functions). SoA chosen over object-style for
  speed and dimension-agnosticism.
- **First version = simplest:** single-species greedy forager — no RNN, no
  genome, no mutation. Matches the forager applet.
- **Output:** fresh npz/HDF5 formats; does not read existing `rep_*.mat` files.

## Full intended scope

Built incrementally, but anticipated in the architecture from the start.

- **Model layers:** pheromone field (reaction-diffusion); predators;
  multi-species / trophic levels; chemoton / internal metabolism; heterogeneous
  environment; evolution / neuroevolution (RNN controllers).
- **Analysis:** information-theoretic measures (semantic info, mutual
  information, transfer entropy); phase / order parameters + transition
  detection; trajectory logging; population statistics (lineages, fitness,
  diversity).
- **Infrastructure:** SLURM parameter sweeps; reproducible config + seeding;
  interactive matplotlib visualization (port `Visualize.m`).
- **Repository:** Sphinx docs + theory guide; pytest suite; `matlab/` verbatim
  archive; example notebooks.

## Extension seams (baked into the skeleton up front)

Cheap to add now, costly to retrofit; make each later layer an add-on rather
than a rewrite.

- Shared `Grid` object (shape, spacing, toroidal neighbour map).
- `Field` protocol (`sample` / `step` / `deposit`); the simulation holds a
  *list* of fields, not one hardcoded resource field.
- Generic `SpatialHash` keyed on any point set (resources or populations).
- Step loop takes a *collection* of populations even if v1 has one.
- `Controller` abstraction reserved for the greedy-vs-RNN split (greedy lives in
  the agent body for v1).

## Open items (pending input before the relevant layers)

- Specific formulation for the chemoton and the multi-species / trophic
  structure.
- npz vs HDF5 default (assumed both supported, npz default).
