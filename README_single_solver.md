# Newton Sand — Single Solver (MPM Only)

**File:** `simulation_newton_sand_single_solver.py`

---

## Changes Made (Confirmed Working)

The following changes were made iteratively to fix particle overflow and particle overlap on the scoop.

### 1. Fix particle overflow out of the container

**Change:** `emit_hi[2]` changed from `0.20` → `0.05`

The container walls top out at z = 0.05 m. Sand was being emitted up to z = 0.20 m — 4× the container height — so the majority of particles spawned outside the container and fell off. Clamping `emit_hi[2]` to `0.05` keeps all initial particles within the container bounds.

---

### 2. Material parameter overhaul (kinetic-sand physics)

All of the following were changed at once to bring parameters in line with Newton's own `example_mpm_granular.py` reference and Klar et al. 2016 (SIGGRAPH sand paper).

| Parameter | Before | After | Reason |
|---|---|---|---|
| `young_modulus` | 5e4 | 1e9 | Original value was 20,000× too soft — any volumetric compression generated almost no restoring pressure, so particles compressed into each other freely. Raised to 1e9 (nearly incompressible for our pressure range). |
| `yield_pressure` | 1e4 | 1e12 | At 1e4 Pa the pile's own weight (~700 Pa) was below yield but the scoop interaction easily exceeded it → plastic flow → merging. Setting to 1e12 keeps the material fully elastic at all realistic pressures; pile shape comes from `friction` alone. |
| `poisson_ratio` | 0.4 | 0.3 | 0.4 caused excessive lateral expansion under vertical load, pushing particles sideways out of the container. 0.3 is the standard value for granular materials. |
| `friction` | 1.4 | 0.8 | Controls the Drucker-Prager angle of repose (`friction = tan(angle)`). The reference uses 0.68 (~34°, dry sand). Kinetic sand is slightly more cohesive (~40°), so 0.8 was chosen. |
| `yield_stress` | 100.0 | 0.0 | Reference uses 0.0; non-zero added unintended cohesion. |
| `hardening` | 1.0 | 0.0 | Reference uses 0.0; hardening=1 was artificially stiffening the yield surface as the material deformed. |
| `air_drag` | 15.0 | 1.0 | Reference uses 1.0; 15.0 caused unnatural particle clustering because drag dominated over inter-particle pressure. |
| `viscosity` | (not set) | 0.02 | Added to give the sand a sluggish, kinetic-sand-like flow rather than instantaneous dry-sand flow. |
| `dilatancy` | (not set) | 0.0 | Wet/kinetic sand does not dilate during shear. Reference uses 0.0. |

---

### 3. Increase shape margin on the scoop

**Change:** `shape_margin` on the scoop mesh changed from `0.5 × particle_radius` → `2.0 × particle_radius`

The shape margin inflates the scoop's SDF surface outward from the mesh triangles. At 0.5× (~0.8 mm), the margin was smaller than one particle radius — thin rim edges of the scoop bowl were effectively invisible to the MPM collider, so particles tunnelled straight through the scoop walls and piled up inside. At 2.0× (~2.5 mm), the collision surface extends far enough that edge geometry is reliably detected.

---

### 4. Reduce particles per cell

**Change:** `particles_per_cell` changed from `3` → `2`

In MPM, all particles within the same grid cell share one velocity sample from the grid. At `particles_per_cell=3` there are 27 particles per 1 cm³ cell — the grid averages their motion and cannot separate sub-cell positions, causing visible overlap. Reducing to 2 (8 particles/cell) means the grid velocity field has fewer particles to average, giving it much better ability to push particles apart through the pressure field.

---

### 5. Increase solver max iterations

**Change:** `max_iterations` changed from `250` → `500`

With the stiffer material (young_modulus=1e9) and a fast-moving scoop compressing sand into the bowl, the implicit linear system is harder to solve. At 250 iterations the solver was not converging cleanly under dynamic scoop loading — resulting in corrupt particle velocities and apparent overlap near the scoop. Doubling to 500 gives the iterative solver enough iterations to reach the tolerance threshold.

---

## Commit 2 Changes

The following changes were made to fix particles penetrating through the scoop walls and to improve grid resolution.

### 1. Add CFL velocity cap (was missing)

**Change:** Added `builder.particle_max_velocity = 0.5 * voxel_size / sim_dt` before `builder.finalize()`

This was missing entirely from the file. Without it, particles near the moving scoop have no speed limit and can travel across multiple grid cells in a single substep — passing straight through thin scoop walls before `project_outside` gets a chance to catch them. The CFL condition caps each particle to half a voxel of travel per substep, which guarantees the SDF query in `project_outside` can always find a tunnelling particle on the correct side of the wall.

At `voxel_size=0.007`, this caps particle velocity at `0.5 × 0.007 × 240 = 0.84 m/s`.

---

### 2. Reduce voxel_size

**Change:** `voxel_size` default changed from `0.01` → `0.007`

A 10 mm grid cell is too coarse for the scoop geometry — thin walls that are 2–3 mm thick span less than one grid cell, so the MPM SDF cannot resolve them and particles slip through. At 7 mm cells the SDF boundary is represented more accurately, making wall detection reliable. The finer grid also tightens the CFL cap (0.84 m/s vs 1.2 m/s), further reducing the chance of tunnelling.

Particle count increases ~3× (the finer grid places more particles in the same emit volume). Pass `--voxel-size 0.01` on the command line to revert if performance is too slow.

---

### 3. Increase shape margin

**Change:** `shape_margin` on the scoop mesh changed from `2.0 × particle_radius` → `3.0 × particle_radius`

Because `particle_radius` is derived from `voxel_size / particles_per_cell * 0.5`, reducing `voxel_size` from 0.01 to 0.007 shrinks the radius from 2.5 mm to 1.75 mm. Keeping the multiplier at 2.0× would have dropped the absolute margin from 5 mm to 3.5 mm. Raising to 3.0× keeps the effective collision surface at ~5.25 mm from the mesh triangles — the same detection distance as before — so scoop rim edges remain visible to the collider.

---

