# pyLEAFS — Task List

Ordered roughly by dependency. Layers after validation are independent add-ons
enabled by the extension seams in the skeleton.

## Foundation

1. **Scaffold pyLEAFS repo (CellularAutomata template).** `pyLEAFS/` package,
   `tests/`, `docs/` (Sphinx), `matlab/` (verbatim originals from the
   `Codebase/` + `Sensors/Codebase/` merge), `pyproject.toml`, `README.md`
   (Install / Quick start / Docs / Tests / License), MIT license, `.gitignore`,
   `git init`.
2. **Build dimension-agnostic core skeleton with extension seams.** Shared
   `Grid` (shape/spacing/toroidal neighbours, D=2/3); `Field` protocol
   (sample/step/deposit) with the simulation holding a *list* of fields; generic
   `SpatialHash` over any point set; multi-population step loop; reserved
   `Controller` abstraction. NumPy struct-of-arrays, `(n,D)` positions,
   Numba-ready, no `@njit` yet.

## First version

3. **Implement v1: simple greedy forager** (no RNN/genome/mutation). Homogeneous
   Poisson birth-death resource field (Ξ→ε, grow/decay), sense-radius
   nearest-resource detection, agent body (kinematics, metabolism
   `ds/dt = -µ₀ + harvest`, reproduce at S_REP, die at s≤0), birth/death
   population step, driver with RNG seeding. Greedy steering in the agent body.
4. **Validate v1 against applet + known limits.** pytest: reproduce applet
   dynamics, phase-plane equilibrium line (N_R/N_Rmax + N_A/N_Amax = 1),
   extinction, seeded reproducibility, D=2 vs D=3 sanity.

## Capabilities

5. **Add observables:** phase plane + transition detection, time-series /
   trajectory logging (npz/HDF5), population statistics, information-theoretic
   measures (semantic info, MI, transfer entropy).
6. **Add run infrastructure:** config-driven runs (versioned parameter sets),
   reproducible seeding, npz/HDF5 output, SLURM array sweep scripts (port
   `leafs_array.slurm`).
7. **Add interactive visualization** (port `Visualize.m`): matplotlib live
   viewer + applet-style phase plane; dimension-aware.

## Model layers (add-ons)

8. **Pheromone field (reaction-diffusion).** `PheromoneField(Field)` on the
   shared Grid: `∂φ/∂t = D∇²φ − decay·φ + deposit`, agent deposit + sense.
9. **Predators + multi-species / trophic levels.** Predator population sensing
   foragers via the generic `SpatialHash`; predator→prey feeding as population
   death; interaction ordering in the multi-population step loop; additional
   trophic levels; predator-prey phase-plane trajectories.
10. **Heterogeneous environment.** Patchy/seasonal fields, gradients,
    terrain/obstacles — region-partitioned `ResourceField` backend with
    spatially-varying Γ (port the leafs applet `gammaField`).
11. **Evolution / neuroevolution (RNN controllers).** `Controller` with
    `GreedyController` + `RNNController` (Elman: W_in/W_rec/tanh/W_out/softmax,
    port `AgentRNN.m`), `Genome` dataclass, `SensorArray` (cone geometry),
    mutation/clone, selection, HOF/epoch/extinction (port `PopulationRNN.m`).
    Pheromone sense becomes an extra RNN input channel.
12. **Chemoton / internal metabolism.** Per-agent metabolic network coupling
    harvested energy to internal state, replacing the simple `ds/dt`.

## Closing

13. **Sphinx docs + theory guide, examples, Numba pass.** API docs + narrative
    theory pages; runnable example notebooks/scripts reproducing key figures and
    the sensor-development study; optional `@njit` pass on hot functions once the
    core is validated.
