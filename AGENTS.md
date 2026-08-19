# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.
- The clone is named `LEAFS`; `pyLEAFS/` inside it is the package, not a project
  root. `pip install -e .` runs at the clone root. Setup, docs build and test
  commands live in the README and in `docs/getting_started.rst`, which must stay
  in step with each other and with `pyproject.toml`.
- The published site is `damiansowinski.com/LEAFS/` (built from `docs/`).
  `damiansowinski.com/pyLEAFS/` is a 404.
- Supported interpreters are Python 3.8 through 3.14, verified by running the
  suite on each. The floor is set by the build backend (`setuptools>=61`,
  `requires-python >=3.7`), not by the code, which needs only
  `numpy>=1.17` (`np.random.default_rng`) and `matplotlib>=3.1`
  (`FuncAnimation(cache_frame_data=...)`); both bounds are declared in
  `pyproject.toml`.
- Docs prose uses a matched pair of spaced regular dashes ( - ), never `---`,
  which Sphinx smartquotes renders as an em dash. `sphinx-build -n` surfaces
  unresolved `:class:` targets because the classes are documented under their
  module paths, not the package root; the one non-nitpick warning comes from the
  `pyLEAFS/spatialhash.py` docstring and predates this note. In a docstring, a
  section header napoleon does not know (`Controls`, say) placed *before*
  `Parameters` silently breaks the whole parameter block into raw `:param:`
  text, so keep custom sections last and check the built HTML, not just the
  warning count.
- Anything that renders must run on the Agg backend: the tests select it at
  import, and an interactive window on the maintainer's machine steals focus.
- A 3d world is far heavier than a 2d one of the same width: the default
  `shape=(10, 10, 10)` holds around 250k resources and booms to thousands of
  agents, and the per-agent Python loop in `Population._harvest` then dominates
  wall-clock time by an order of magnitude over drawing. Profile the step loop,
  not the viewer, and use a smaller `shape` or `Xi` for quick 3d checks.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
