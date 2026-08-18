# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.
- The clone is named `LEAFS`; `pyLEAFS/` inside it is the package, not a project
  root. `pip install -e .` runs at the clone root. Setup, docs build and test
  commands live in the README and in `docs/getting_started.rst`, which must stay
  in step with each other and with `pyproject.toml`.
- The published site is `damiansowinski.com/LEAFS/` (built from `docs/`).
  `damiansowinski.com/pyLEAFS/` is a 404; `pyproject.toml`'s `Documentation`
  URL still points there.
- Supported interpreters are Python 3.8 through 3.14, verified by running the
  suite on each. The floor is set by the build backend (`setuptools>=61`,
  `requires-python >=3.7`), not by the code, which needs only
  `numpy>=1.17` (`np.random.default_rng`) and `matplotlib>=3.1`
  (`FuncAnimation(cache_frame_data=...)`). Neither lower bound is declared.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
