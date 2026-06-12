# pyLEAFS — Working Instructions

Process rules for development in this folder.

## Process

- **Always ask before writing code.** Describe the proposed code (what files,
  what each does, the approach) and wait for explicit permission. "Do it,"
  "proceed," "go ahead," "make it so," "let's try it" grant permission to write
  the code just described.
- **Make targeted edits, not complete rewrites.**
- **Leave design questions to Damian.** Present options and trade-offs rather
  than deciding.
- **No stylistic embellishments** unless specifically asked for.
- **When asked to move a file, move it** — do not rewrite it at the new location.
- **Update `journal.txt`** (in the project root) with dated summaries of work
  done.
- When fixing a bug, do not claim it is fixed — ask whether the implementation
  works.

## Design contract (decided)

- Mirror the `EternalTime/CellularAutomata` repo layout.
- Dimension-agnostic core (D = 2 or 3 via parameter; `(n, D)` position arrays).
- NumPy struct-of-arrays; Numba-ready but no `@njit` until the core is
  validated.
- First version: single-species greedy forager only — no RNN, genome, or
  mutation.
- Output to fresh npz/HDF5; do not read the legacy `rep_*.mat` files.

## Extension seams to preserve

Any new code must keep these intact so later layers stay add-ons:

- A shared `Grid` referenced by all fields.
- A `Field` protocol; the simulation iterates a *list* of fields.
- A generic `SpatialHash` usable over resources or populations.
- A step loop over a *collection* of populations.
- A reserved `Controller` abstraction for the greedy-vs-RNN split.
