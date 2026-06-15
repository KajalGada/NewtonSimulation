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

## What This File Does

This is a single-solver MPM simulation of kinetic sand being scooped by a robot end-effector. It is the target environment for RL training in Isaac Lab.

- **No MuJoCo, no UR5 arm.** The scoop is a standalone kinematic rigid body.
- **No impedance or velocity control.** The scoop pose is replayed frame-by-frame from a pre-recorded dataset of world-frame Cartesian poses (`demo_20260518_143752_scoop_poses.npz`).
- **All particle physics** (gravity, pressure, friction, collision) is handled entirely by Newton's `SolverImplicitMPM`.

---

## Why Single Solver (Not Dual Solver)

The previous dual-solver design ran MuJoCo for rigid-body dynamics and Newton MPM for sand, coupling them at each timestep. This was dropped for RL because:

- RL training uses **position control**, not impedance/velocity control, so MuJoCo's joint-space solver adds no value.
- The dual-solver coupling introduced a rigid-rigid collision ambiguity (material projection thresholds, two-way contact) that was hard to stabilize.
- In Isaac Lab, the robot policy outputs target poses directly, so the sim only needs to answer: "given this scoop pose, how does the sand respond?"

The single solver eliminates all of that. The scoop pose is simply written into `body_q` every substep; the MPM solver treats it as a moving SDF boundary.

---

## Architecture

```
dataset .npz  →  _get_scoop_pose(t)  →  body_q (kinematic)
                                               ↓
                          SolverImplicitMPM.step()   (sand physics)
                                               ↓
                          project_outside()           (catch tunnelled particles)
```

The substep loop runs `sim_substeps=4` times per rendered frame at `fps=60`, giving a physics timestep of `sim_dt = 1/(60×4) ≈ 4.2 ms`.

---

## Scene Setup

### Container

A 35 cm × 35 cm open-top box made of 5 static axis-aligned box shapes:

| Shape | Purpose |
|---|---|
| Floor | 35×35×2 cm, z-centre = 1 cm |
| Front/back walls | 35×35×5 cm half-widths in Y |
| Left/right walls | 35×35×5 cm half-widths in X |

Wall top is at **z = 0.05 m**. Sand is emitted from z = 0.03 to z = 0.05, so it fills at most 2 cm of headroom — particles stay inside on initialisation.

All box shapes use `gap=0.01` (ShapeConfig) so the MPM collider adds a 1 cm buffer zone at the inner surfaces.

### Scoop Mesh

Loaded from `ur_urdf/newscoop.stl` (millimetre units → scaled by 0.001).

**Orientation fix:** the STL is stored in URDF local space where the bowl opens in the +Y direction. The URDF collision origin specifies `rpy="-1.5707963267948966 0 0"` to correct this. Since we load the STL directly (no URDF parser), the same -90° rotation around X is baked into the vertices manually:

```python
_Rx = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float32)  # Rx(-90°)
_vertices = _vertices @ _Rx.T
```

This maps Y→Z so the bowl opens upward in world space.

**Shape margin:** set to `2.0 × particle_radius`. This inflates the scoop's SDF surface outward, ensuring thin rim edges are detected before a fast-moving particle can tunnel through them. At 0.5× (the earlier value) the ~1 mm margin was smaller than one particle radius and edges were effectively invisible to the collider.

---

## Sand Particles

Particles are placed on a regular grid within the emit volume:

```
emit_lo = [-0.15, -0.35, 0.03]   # just above the container floor
emit_hi = [ 0.15, -0.05, 0.05]   # at container wall height
```

`particles_per_cell = 2` means 2 sample points per voxel length per axis → 8 particles per 1 cm³ grid cell. This was deliberately reduced from 3 (27 particles/cell) because:

- In MPM, all particles in the same grid cell share one velocity field. At 27 particles/cell, the solver averages over too many particles → sub-cell overlap cannot be separated.
- At 8 particles/cell the grid can better differentiate particle motion within a cell, significantly reducing visible overlap.

`particle_radius` is derived as `voxel_size / particles_per_cell * 0.5`, so it automatically scales with grid resolution.

---

## Material Parameters

All parameters are exposed as argparse flags and routed to the solver at startup via a generic loop that checks both `SolverImplicitMPM.Config` and `model.mpm` attributes. `air_drag` is explicitly re-set after the loop because it lives on the Config only, not on `model.mpm`.

### Reference

Parameters are grounded in two sources:

1. **Newton's own `example_mpm_granular.py`** — the canonical reference for granular MPM in this codebase.
2. **Klar et al., "Drucker-Prager Elastoplasticity for Sand Animation," SIGGRAPH 2016** — the primary academic reference for MPM sand.

### Parameter Table

| Parameter | Value | Why |
|---|---|---|
| `young_modulus` | 1e9 Pa | Grains are nearly incompressible (real quartz ~7×10¹⁰ Pa). We use 1e9 rather than the reference's 1e15 because the moving scoop creates high local pressures that make 1e15 numerically unstable (the implicit solver diverges in 500 iterations → corrupt velocities → apparent overlap). 1e9 is still 20,000× stiffer than the original 5e4, which was the root cause of particle interpenetration. |
| `yield_pressure` | 1e12 Pa | Keeps the material in the elastic regime for all physically realistic pressures. Pile shape (angle of repose) is controlled by `friction` alone, not by plastic pressure yield. The original value of 50 Pa meant the material's own weight (~700 Pa) exceeded yield constantly → continuous plastic flow → particles merged. |
| `poisson_ratio` | 0.3 | Standard for granular materials (Klar 2016 uses 0.3). The previous value of 0.4 caused excessive lateral expansion under vertical compression, pushing particles sideways out of the container. |
| `friction` | 0.8 | Controls the Drucker-Prager yield surface = the macroscopic angle of repose. `friction = tan(angle)`. Dry sand ≈ 0.68 (34°, Newton reference). Kinetic sand (silicone-coated) ≈ 0.84–1.0 (40–45°). We use 0.8 as a slightly cohesive kinetic-sand value. |
| `damping` | 0.0 | `alpha = dt/(dt+damping)` multiplies elastic stress each step. At damping=450 (original), alpha≈9e-6 → elastic stress is zeroed every step → no shear resistance to hold a pile. Must be 0. |
| `viscosity` | 0.02 | Adds a viscous stress component proportional to strain rate. Makes sand flow sluggishly rather than instantly, giving kinetic-sand character. Not present in the original file. |
| `dilatancy` | 0.0 | Wet/kinetic sand does not dilate during shear (dry sand does slightly). Reference uses 0.0. |
| `yield_stress` | 0.0 | Reference uses 0.0. Non-zero values add cohesion that was not in the original intent. |
| `hardening` | 0.0 | Reference uses 0.0. The original value of 1.0 was artificially stiffening the yield surface as the material deformed. |
| `air_drag` | 1.0 | Reference uses 1.0. The original value of 15.0 caused unnatural clustering as particle drag dominated over inter-particle pressure. |
| `density` | 1400 kg/m³ | Reasonable for damp/kinetic sand (dry quartz ~1600, kinetic sand product ~1200–1500). |

### CFL Velocity Cap

```python
builder.particle_max_velocity = 0.5 * voxel_size / sim_dt
```

This caps particle velocity so no particle can travel more than half a voxel per timestep. Without this, fast-moving particles near the scoop rim tunnel through thin mesh walls before the next `project_outside` call.

---

## Collision Handling

### MPM Collider (mesh SDF)

`SolverImplicitMPM` builds an SDF from each shape's geometry. At each substep, particles near a surface have their grid velocities modified to not penetrate the surface (`collider_velocity_mode = "backward"` — computes surface velocity from the difference in `body_q` between the current and previous substep).

### `project_outside`

Called **after** each MPM step (not before). It finds particles that have tunnelled into a solid during the momentum solve and ejects them:

```python
self.mpm_solver.project_outside(
    self.state_0, self.state_0, self.sim_dt,
    gap=self.mpm_solver.voxel_size * 2.0,
)
```

`gap = 2 × voxel_size = 20 mm` gives a search radius large enough to catch particles that entered a thin wall at `particle_max_velocity` and need to be found on the far side of the wall within one substep.

Running `project_outside` before the MPM step (the original ordering) caused double-correction: particles near surfaces were ejected by `project_outside`, then the MPM velocity solve pushed them back → they clustered at the surface.

---

## Known Limitations

### Scoop Does Not Collide With Container Walls

The scoop's pose is assigned directly from the dataset every substep. There is no physics solver enforcing rigid-rigid contact between the scoop and the container walls. If the dataset trajectory brings the scoop into a wall, it passes through. This is an architectural limitation of the kinematic single-solver design. Fixes:

- **Preprocess the dataset** to clamp any poses that would intersect the container.
- **For RL:** the policy learns to avoid walls from its own reward signal; this issue disappears when the scoop pose comes from the policy rather than a replay.

### Sub-Voxel Particle Overlap

MPM separates particles through the grid pressure field. All particles within the same grid cell share one velocity sample. Overlap at scales smaller than `voxel_size` (1 cm) cannot be prevented by the solver regardless of material stiffness. Visible improvement requires either reducing `voxel_size` (e.g. 0.007 m → ~3× more expensive) or reducing `particles_per_cell`.

### Open Mesh SDF Ambiguity

The scoop bowl is open at the top. MPM's SDF query for an open mesh can misclassify particles inside the bowl cavity as "inside solid" depending on local face normals. `project_outside` may occasionally eject particles from the bowl interior. If this causes sand to not collect in the scoop, the fix is to close the top of the mesh (add a cap face in the STL).
