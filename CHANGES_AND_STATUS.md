# Newton Simulation — Changes, Fixes, and Open Issues

## Original Problem: No Two-Way Coupling

The original simulation had **no two-way coupling between the sand and the robot**.

`setup_collider()` was called without a `body_mass` argument:

```python
# Original — kinematic only, sand cannot push back
self.mpm_solver.setup_collider(
    body_q=self.state_0.body_q,
)
```

Without a mass array, the MPM solver treats all bodies as **kinematic**: the
scoop moves through the sand and displaces particles, but the sand exerts zero
reaction force on the robot. The joint torques were entirely unaffected by sand
resistance. There was also no wrench readout — no way to observe what force the
sand was applying to the end-effector at all.

This meant:
- The robot arm trajectory was not physically influenced by the sand.
- No force/torque signal was available for downstream control or RL reward.
- The simulation was essentially a one-way animation: robot moves, sand reacts,
  nothing feeds back.

---

## What Was Changed and Why

### 1. Two-way coupling activated (`setup_collider` with real body mass)

```python
# Updated — real mass enables full two-way coupling
self.mpm_solver.setup_collider(
    body_mass=self.model.body_mass,
    body_q=self.state_0.body_q,
)
```

Passing `body_mass` activates the **Delassus operator** inside the implicit MPM
Gauss-Seidel solve. The solver now accounts for body inertia when distributing
contact impulses, so the sand physically resists the scoop and the resulting
forces propagate back to the robot's joints.

---

### 2. GPU kernel to apply sand forces back to the robot every substep

A Warp GPU kernel (`_compute_body_forces`) was added to reduce the per-grid-node
MPM impulses into a 6-DOF wrench and accumulate it into `state.body_f` before
each robot solve:

```python
@wp.kernel
def _compute_body_forces(dt, collider_ids, collider_impulses, collider_impulse_pos,
                          body_ids, body_q, body_com, body_f):
    i = wp.tid()
    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return
        f_world = collider_impulses[i] / dt
        r = collider_impulse_pos[i] - wp.transform_point(body_q[body_index], body_com[body_index])
        wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))
```

This runs every substep so the robot solver sees the sand force at each
integration step, not once per frame.

---

### 3. Constant ~236 N wrench bug — impulse filtering

When a `[WRENCH]` readout was first added, it reported a constant ~236 N
regardless of whether the scoop was in the sand or not.

**Root cause:** `collect_collider_impulses()` returns impulses from *every*
collider in the scene — the ground plane, all four container walls, and the
scoop. Summing them all gives the total sand weight (~236 N), a constant.

**Fix:** use the third return value (`collider_ids`) together with
`mpm_solver.collider_body_index` to build a mask `body_index >= 0`, excluding
the static scene geometry (ground/walls map to body index −1):

```python
body_np = self._collider_body_idx[ids_np]
mask    = body_np >= 0          # True only for the scoop, not ground/walls
self._wrench_f += imp_np[mask].sum(axis=0) / self.sim_dt
```

After the fix, the wrench reads ~0 N when the scoop is above the sand and
0.2–0.6 N when it is immersed — physically sensible.

---

### 4. Wrong scoop body index (body 12 instead of 11)

Diagnostic code used `model.body_count - 1 = 12` to identify the scoop. Body 12
is actually the "base" fixed frame at world origin (z = 0 always), not the
scoop.

**Fix:** derive the scoop index at runtime from the authoritative MPM collider
map:
```python
int(max(b for b in self._collider_body_idx if b >= 0))  # → 11
```
`MPM collider_body_index = [-1, 1, 2, 3, 4, 5, 6, 7, 11]` confirmed scoop = body 11.

---

### 5. Scoop mass set to physically correct value

The scoop link `<mass>` in the URDF was not set to a measured value. The scoop
was modelled in Fusion 360 with ABS plastic material and found to weigh
**0.11 kg (110 g)**. Both URDFs were updated.

---

### 6. High-polygon scoop mesh replaced (14,086 → 1,084 triangles)

`ur5_scoop.stl` had 14,086 triangles. A remeshed version `newscoop.stl` (1,084
triangles) was introduced in `ur5_with_scoop_v2.urdf`. The simulation now loads
`ur5_with_scoop_v2.urdf`. The mesh reduction has minimal impact on overall FPS
because the MPM Gauss-Seidel solve dominates at ~85–90% of wall time, but it
reduces SDF rasterization by ~1–2 ms per substep.

---

### 7. Floating-particle mitigations

Two parameters were tuned to reduce particles hovering above the sand pile after
leaving the scoop:

- **`shape_margin`** reduced from `2.0×` to `0.5×` particle radius — cuts the
  standoff gap at the scoop surface, so particles sit closer to the mesh.
- **`project_outside` gap** reduced from `voxel_size × 0.25` to
  `voxel_size × 0.1` (~1 mm) — smaller outward projection when the scoop lifts
  means less upward kick imparted to edge particles.
- **`--air-drag`** default raised from 6.0 to 10.0 — higher viscous damping
  absorbs residual hover velocity faster without meaningfully thickening the
  bulk sand behaviour.

---

### 8. API rename for Newton 1.2.1

The project was updated from Newton 1.4.0.dev0 (local editable source) to
Newton 1.2.1 (latest PyPI release, Warp 1.13.0). In 1.2.1 the `Control` object
renamed `joint_target_q` → `joint_target_pos`. All four call sites were updated.

---

## Validation

| Check | Result |
|-------|--------|
| Wrench above sand | `\|F\| ≈ 0 N` — correct |
| Wrench during scooping | 0.2–0.6 N — physically consistent with ~100 g sand load at this scale |
| Scoop z vs floor | Min observed z = 0.11 m; floor top = 0.02 m — no penetration |
| FPS on RTX 3060 | ~3.6–4.0 FPS (250 ms/frame, 4 substeps × ~55 ms MPM) |
| Newton 1.2.1 boot | Clean, all modules load from cache |

---

## Open Issues / Remaining Work

### A. Container floor collision not resolving

Some particles pass through or ignore the container floor bottom surface. The
container box shapes use `ShapeConfig(gap=0.01)`, but the `project_outside` gap
that was just reduced to `voxel_size × 0.1` may now be too small to prevent
fast-moving particles from tunnelling through the floor (which has a different
normal direction than the scoop).

**Suggested investigation:**
- Log minimum particle z each frame to confirm particles are genuinely below
  `wall_thickness = 0.02 m` (may still be a visual/perspective artifact).
- Consider applying a shape-specific gap for the floor shapes at the builder
  level rather than a global `project_outside` gap.
- Test with `project_outside` gap restored to `voxel_size × 0.25` to check
  whether the floor issue existed before the reduction.

### B. Particles hovering above the pile after leaving the scoop

When the scoop tilts and particles cascade off the rim, some float briefly above
the settled pile before sinking.

**Likely causes:**
1. `project_outside` still imparts a small outward (sometimes upward) velocity
   at the curved scoop edge even at the reduced gap.
2. The MPM background grid at `voxel_size = 0.01 m` (~6× particle radius) creates
   a coarse pressure field that can momentarily support particles above the true
   surface.
3. The APIC transfer scheme conserves angular momentum per-particle, which can
   give falling particles a small rotational velocity component.

**Suggested investigation:**
- Try `--air-drag 14` or `--air-drag 18` from the command line (no code change)
  and see if float duration decreases proportionally.
- Try `--transfer-scheme pic` (less angular momentum conservation) and compare.
- Reduce `voxel_size` from 0.01 to 0.008 m for a finer grid — directly reduces
  the surface "cushion" but increases MPM solve time by ~(10/8)³ ≈ 2×.



*Branch: `optimization` — push to remote for RTX 5080 testing.*
