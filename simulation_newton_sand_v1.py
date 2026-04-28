# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Newton MPM simulation of kinetic sand with a UR5 + scoop end-effector.
#
# Structure follows newton/examples/mpm/example_mpm_granular.py.
# Robot loading follows newton/examples/robot/example_robot_panda_hydro.py.
#
# Material parameters are tuned from granular_object.py to reproduce
# kinetic-sand behaviour: deformable, high-friction, heavily-damped,
# low yield-strength so the bulk flows under disturbance but holds its
# shape at rest.
#
# Robot control: PD joint position targets via SolverFeatherstone.
# Joint angles are interpolated between waypoints each frame; Featherstone
# integrates dynamics and applies contact forces so the robot collides with
# the box.  SolverImplicitMPM handles particle↔rigid collision separately.

import sys

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM, SolverMuJoCo

# ---------------------------------------------------------------------------
# UR5e waypoints — joint angles computed via numerical IK.
#   joint order: shoulder_pan, shoulder_lift, elbow,
#                wrist_1, wrist_2, wrist_3
#
# Robot base sits at (0.5, 0, 0).  Sand pile spawns at
#   x, y ∈ [-0.15, 0.15],  z ∈ [0.02, 0.20]  and settles to ~0.10 m tall.
#
# emit_hi z is capped at 0.20 m so the forearm (body centre z ≈ 0.29 m in
# _Q_ABOVE) stays above the particle spawn volume.  Starting with any link
# inside the initial particle cloud causes the MPM contact response to blow
# up on the very first step.
#
# EE targets used for IK (world frame):
#   above_sand : (0.0,  0.00,  0.30)  — hover above pile while sand settles
#   enter_sand : (0.0, -0.13,  0.05)  — descend into pile, y = -0.13
#   mid_sand   : (0.0,  0.00,  0.05)  — midpoint of sweep
#   exit_sand  : (0.0, +0.13,  0.05)  — far edge of pile, y = +0.13
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Joint angles: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
#
# EE-only sweep strategy:
#   - shoulder_lift and elbow are FIXED for both in-sand poses so the forearm
#     stays at a constant height (above the sand surface) throughout the sweep.
#   - Only shoulder_pan changes between SIDE_A and SIDE_B, rotating the arm
#     around the base Z-axis so only the scoop traces an arc through the sand.
# ---------------------------------------------------------------------------
_W3 = -np.pi / 2   # scoop orientation (wrist_3)

# Hover above sand
_Q_ABOVE = np.array([-0.27, -1.50,  1.70,  np.pi / 2,  np.pi / 2,  _W3], dtype=np.float32)

# Scooping operation — U-shaped trajectory.
# wrist_2 = 0 keeps the scoop face parallel to the ground during scooping
# (vs wrist_2 = π/2 used in the hover which tilts the scoop for travel).
#
# Box bounds: ±0.175 m in x and y (box_width = box_depth = 0.35 m).
# Robot base is at x = 0.5 m; EE_y ≈ −L·sin(shoulder_pan) where L ≈ 0.45 m.
# shoulder_pan must stay within ±0.38 rad to keep EE_y inside the box.
# Previous value of −0.55 rad pushed EE_y to ≈ +0.37 m (outside the wall).
#
# shoulder_pan: ±0.20 rad keeps EE_y ≈ ±0.09 m — well inside both walls.
# shoulder_lift/elbow increased slightly (−0.55/1.10 vs −0.50/1.00) to extend
# reach in x so the EE lands over the sand pile with wrist_2 = 0.
#
# Entry / exit: shoulder_lift=-0.85, elbow=1.30  →  EE above the sand surface.
# In-sand:      shoulder_lift=-0.55, elbow=1.10  →  EE at sand depth.
_Q_ENTRY = np.array([ -0.55, -0.8,  0.80,  np.pi / 2,  0.0,  _W3], dtype=np.float32)  # above sand, right
_Q_IN_R  = np.array([ 0.20, -0.8,  0.80,  np.pi / 2,  np.pi / 4,  _W3], dtype=np.float32)  # in sand,    right
_Q_IN_L  = np.array([-0.55, -0.8,  0.80,  np.pi / 2,  np.pi / 2,  _W3], dtype=np.float32)  # in sand,    left
_Q_EXIT  = np.array([-0.20, -0.8,  0.80,  np.pi / 2,  0.0,  _W3], dtype=np.float32)  # above sand, left

# Waypoint list: (target_joint_angles, duration_seconds)
# Each cycle traces one U-shaped scoop:
#   ENTRY  → IN_R  : descend right side of U into sand
#   IN_R   → IN_L  : traverse left through sand (bottom of U)
#   IN_L   → EXIT  : ascend left side of U out of sand
#   EXIT   → ENTRY : reset swing above the sand (in air) for next pass
_WAYPOINTS = [
    (_Q_ABOVE,  2.5),  # hover — let sand settle under gravity
    (_Q_ENTRY,  1.5),  # approach: above sand on right, scoop parallel to ground
    (_Q_IN_R,   1.5),  # descend right side of U into sand
    (_Q_IN_L,   3.0),  # traverse left through sand (bottom of U)
    (_Q_EXIT,   1.5),  # ascend left side of U out of sand
    (_Q_ENTRY,  2.0),  # reset: swing back above right (through air)
    (_Q_IN_R,   1.5),  # second pass: descend right
    (_Q_IN_L,   3.0),  # traverse left
    (_Q_EXIT,   1.5),  # ascend left
    (_Q_ENTRY,  2.0),  # reset: swing back above right
    (_Q_ABOVE,  1.5),  # lift out — loop repeats
]


class Example:
    def __init__(self, viewer, args):
        # ------------------------------------------------------------------
        # Simulation timing
        # ------------------------------------------------------------------
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # ------------------------------------------------------------------
        # Scene geometry
        # ------------------------------------------------------------------
        builder = newton.ModelBuilder()

        # Must be called before any bodies / particles are added
        SolverImplicitMPM.register_custom_attributes(builder)

        # ---- UR5 + scoop robot -----------------------------------------
        # Placed at (0.5, 0, 0) with identity orientation.
        # shoulder_pan = π (set in initial joint_q below) rotates the arm
        # to face the sand pile at origin.
        _URDF_PATH = "/home/gmr/Downloads/NewtonSimulation/ur_urdf/ur5_with_scoop.urdf"
        builder.add_urdf(
            _URDF_PATH,
            xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi)),
            floating=False,
            enable_self_collisions=False,
        )

        # URDF shapes default to shape_margin=0, which gives the MPM collider
        # zero thickness.  The grid-node activation condition is:
        #   sdf = distance - thickness < 0.5 * voxel_size
        # so a node activates when distance < thickness + 0.5 * voxel_size.
        # Setting thickness = voxel_size ensures the zone extends 1.5 * voxel_size,
        # covering the full B-spline support radius so every particle near the
        # wall sees at least one activated collider node.
        for i in range(builder.shape_count):
            builder.shape_margin[i] = args.voxel_size

        # UR5 contributes 6 revolute DOFs (fixed joints have 0 DOFs).
        # They are the first (and only) entries in builder.joint_q.
        self.ur5e_dof = 6
        builder.joint_q[:self.ur5e_dof] = _Q_ABOVE.tolist()

        # PD gains — must be set on builder before finalize (like panda_hydro example).
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 2000.0   # Nm/rad — position stiffness
            builder.joint_target_kd[i] =  100.0   # Nms/rad — velocity damping

        # ---- Optional extra collider in the scene ----------------------
        self.collider = args.collider
        if self.collider == "concave":
            extents = (1.0, 2.0, 0.25)
            left_xform = wp.transform(
                wp.vec3(-0.7, 0.0, 0.8),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 4.0),
            )
            right_xform = wp.transform(
                wp.vec3(0.7, 0.0, 0.8),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -np.pi / 4.0),
            )
            for xform in (left_xform, right_xform):
                builder.add_shape_box(
                    body=-1,
                    cfg=newton.ModelBuilder.ShapeConfig(mu=0.1, density=0.0),
                    xform=xform,
                    hx=extents[0],
                    hy=extents[1],
                    hz=extents[2],
                )
        elif self.collider != "none":
            extents = (0.5, 2.0, 0.8)
            if self.collider == "cube":
                xform = wp.transform(wp.vec3(0.75, 0.0, 0.8), wp.quat_identity())
            elif self.collider == "wedge":
                extents = (0.5, 2.0, 0.5)
                xform = wp.transform(
                    wp.vec3(0.1, 0.0, 0.5),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 4.0),
                )
            else:
                xform = wp.transform(wp.vec3(0.0, 0.0, 0.9), wp.quat_identity())
            builder.add_shape_box(
                body=-1,
                cfg=newton.ModelBuilder.ShapeConfig(mu=0.1, density=0.0),
                xform=xform,
                hx=extents[0],
                hy=extents[1],
                hz=extents[2],
            )

        # Ground plane — moderate friction so sand piles rather than slides
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        # ---- Static container box — sand falls into this ---------------
        box_width      = 0.35
        box_depth      = 0.35
        box_height     = 0.05
        wall_thickness = 0.02  # must be >= voxel_size (0.06) so the MPM grid can resolve the wall

        box_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.6,
            gap=0.01,  # must be >= particle radius (~0.01 m at default voxel_size=0.06)
        )

        # Bottom
        builder.add_shape_box(
            body=-1, cfg=box_cfg,
            xform=wp.transform(wp.vec3(0.0, 0.0, wall_thickness * 0.5), wp.quat_identity()),
            hx=box_width * 0.5, hy=box_depth * 0.5, hz=wall_thickness * 0.5,
        )
        # Front wall (−y)
        builder.add_shape_box(
            body=-1, cfg=box_cfg,
            xform=wp.transform(wp.vec3(0.0, -box_depth * 0.5, box_height * 0.5), wp.quat_identity()),
            hx=box_width * 0.5, hy=wall_thickness * 0.5, hz=box_height * 0.5,
        )
        # Back wall (+y)
        builder.add_shape_box(
            body=-1, cfg=box_cfg,
            xform=wp.transform(wp.vec3(0.0, box_depth * 0.5, box_height * 0.5), wp.quat_identity()),
            hx=box_width * 0.5, hy=wall_thickness * 0.5, hz=box_height * 0.5,
        )
        # Left wall (−x)
        builder.add_shape_box(
            body=-1, cfg=box_cfg,
            xform=wp.transform(wp.vec3(-box_width * 0.5, 0.0, box_height * 0.5), wp.quat_identity()),
            hx=wall_thickness * 0.5, hy=box_depth * 0.5, hz=box_height * 0.5,
        )
        # Right wall (+x)
        builder.add_shape_box(
            body=-1, cfg=box_cfg,
            xform=wp.transform(wp.vec3(box_width * 0.5, 0.0, box_height * 0.5), wp.quat_identity()),
            hx=wall_thickness * 0.5, hy=box_depth * 0.5, hz=box_height * 0.5,
        )

        # ---- Sand particles --------------------------------------------
        Example.emit_particles(builder, args)

        # ------------------------------------------------------------------
        # Model & MPM solver
        # ------------------------------------------------------------------
        self.model = builder.finalize()
        self.model.set_gravity(args.gravity)

        mpm_options = SolverImplicitMPM.Config()
        for key in vars(args):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(args, key))
            if hasattr(self.model.mpm, key):
                getattr(self.model.mpm, key).fill_(getattr(args, key))

        mpm_options.collider_velocity_mode = "finite_difference"

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Initialise body_q from the builder's starting pose.
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # ---- MPM solver (particles ↔ robot/box) -------------------------
        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )

        # ---- MuJoCo solver (robot dynamics, built-in contact detection) --
        self.robot_solver = SolverMuJoCo(self.model)

        # ---- PD joint position control -----------------------------------
        self.control = self.model.control()

        # Seed control targets to the starting pose so there is no impulse
        # on the very first substep.
        pos_np = self.control.joint_target_pos.numpy()
        pos_np[:self.ur5e_dof] = _Q_ABOVE
        self.control.joint_target_pos.assign(pos_np)

        # ------------------------------------------------------------------
        # Waypoint tracking (joint-space interpolation)
        # ------------------------------------------------------------------
        self._wp_idx = 0
        self._wp_elapsed = 0.0   # seconds spent in current waypoint

        # ------------------------------------------------------------------
        # Viewer setup
        # ------------------------------------------------------------------
        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self.viewer.show_particles = True

        self.capture()

    # ------------------------------------------------------------------
    # Robot control
    # ------------------------------------------------------------------
    def _update_robot_target(self):
        """Interpolate waypoints and write joint position targets for PD control.

        Called once per frame (before CUDA graph launch) so the CPU-side array
        update is visible to the GPU kernels inside the graph.
        """
        wp_target, wp_duration = _WAYPOINTS[self._wp_idx]
        prev_idx = (self._wp_idx - 1) % len(_WAYPOINTS)
        wp_prev, _ = _WAYPOINTS[prev_idx]

        t = min(self._wp_elapsed / wp_duration, 1.0)
        q_target = (1.0 - t) * wp_prev + t * wp_target

        pos_np = self.control.joint_target_pos.numpy()
        pos_np[:self.ur5e_dof] = q_target
        self.control.joint_target_pos.assign(pos_np)

        self._wp_elapsed += self.frame_dt
        if self._wp_elapsed >= wp_duration:
            self._wp_idx = (self._wp_idx + 1) % len(_WAYPOINTS)
            self._wp_elapsed = 0.0

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    def capture(self):
        self.robot_graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate_robot()
            self.robot_graph = capture.graph

        self.sand_graph = None
        if wp.get_device().is_cuda and self.mpm_solver.grid_type == "fixed":
            with wp.ScopedCapture() as capture:
                self.simulate_sand()
            self.sand_graph = capture.graph

    def simulate_robot(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.robot_solver.step(self.state_0, self.state_1, self.control, contacts=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_sand(self):
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

    def step(self):
        # Update PD targets on CPU before the (possibly captured) GPU graph
        self._update_robot_target()

        if self.robot_graph:
            wp.capture_launch(self.robot_graph)
        else:
            self.simulate_robot()

        if self.sand_graph:
            wp.capture_launch(self.sand_graph)
        else:
            self.simulate_sand()

        self.sim_time += self.frame_dt

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def render_ui(self, imgui):
        pass

    # ------------------------------------------------------------------
    # Particles
    # ------------------------------------------------------------------
    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        density = args.density
        voxel_size = args.voxel_size
        particles_per_cell = args.particles_per_cell

        particle_lo = np.array(args.emit_lo, dtype=np.float32)
        particle_hi = np.array(args.emit_hi, dtype=np.float32)
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        cell_size = (particle_hi - particle_lo) / particle_res
        cell_volume = np.prod(cell_size)
        radius = float(np.max(cell_size) * 0.5)
        mass = cell_volume * density

        print(f"[INFO] particle_res = {particle_res.tolist()}")
        print(f"[INFO] total particles ≈ {(particle_res[0]+1)*(particle_res[1]+1)*(particle_res[2]+1)}")
        print(f"[INFO] particle radius ≈ {radius:.5f} m")
        print(f"[INFO] particle mass   ≈ {mass:.6f} kg")

        builder.add_particle_grid(
            pos=wp.vec3(particle_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=particle_res[0] + 1,
            dim_y=particle_res[1] + 1,
            dim_z=particle_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=args.initial_jitter * radius,
            radius_mean=radius,
        )

    # ------------------------------------------------------------------
    # Argument parser
    # ------------------------------------------------------------------
    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # Waypoint cycle is 2.5+2.0+2.5+2.5+2.0 = 11.5 s.
        # Override the framework default of 100 frames so USD recording captures
        # at least one full robot sweep (11.5 s × 60 fps = 690 frames).
        parser.set_defaults(num_frames=1000)

        # Scene
        parser.add_argument(
            "--collider",
            default="none",
            choices=["cube", "wedge", "concave", "none"],
            type=str,
        )
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])

        # Timing
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=4)

        # Particle volume — compact initial pile matching granular_object.py
        parser.add_argument("--emit-lo", type=float, nargs=3, default=[-0.15, -0.15, 0.02])
        # z capped at 0.20 so no robot link overlaps the spawn volume at startup
        parser.add_argument("--emit-hi", type=float, nargs=3, default=[ 0.15,  0.15, 0.20])
        parser.add_argument("--particles-per-cell", type=int, default=3)
        parser.add_argument("--initial-jitter", type=float, default=0.5)

        # Grid / solver
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.01)
        parser.add_argument(
            "--grid-type", "-gt", type=str,
            default="sparse", choices=["sparse", "fixed", "dense"],
        )
        parser.add_argument("--grid-padding", "-gp", type=int, default=0)
        parser.add_argument("--max-active-cell-count", "-mac", type=int, default=-1)
        parser.add_argument(
            "--solver", "-s", type=str,
            default="gauss-seidel",
            choices=["gauss-seidel", "jacobi", "cg", "cg+jacobi", "cg+gauss-seidel"],
        )
        parser.add_argument(
            "--transfer-scheme", "-ts", type=str,
            default="apic", choices=["apic", "pic"],
        )
        parser.add_argument("--strain-basis", "-sb", type=str, default="P0")
        parser.add_argument("--collider-basis", "-cb", type=str, default="Q1")
        parser.add_argument("--max-iterations", "-it", type=int, default=250)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
        parser.add_argument(
            "--collider-velocity-mode", "-cvm", type=str,
            default="finite_difference",
            choices=["instantaneous", "finite_difference"],
            help="finite_difference uses (body_q-body_q_prev)/dt — required for FK-driven robots",
        )

        # ------------------------------------------------------------------
        # Kinetic-sand material parameters (from granular_object.py)
        #
        # Kinetic sand differs from dry sand primarily in three ways:
        #   1. High inter-particle friction  (friction = 1.2)  — grains lock
        #      together under compression but release easily under shear.
        #   2. Heavy viscous damping (damping = 800)  — kills elastic rebound
        #      so the material "flows" rather than bounces.
        #   3. Low yield thresholds (yield_pressure = 300, yield_stress = 150)
        #      — the skeleton yields at small loads, giving the characteristic
        #      soft, mouldable feel.
        # ------------------------------------------------------------------
        parser.add_argument("--density",            type=float, default=1100.0)
        parser.add_argument("--air-drag",           type=float, default=1.0)
        parser.add_argument("--critical-fraction",  "-cf", type=float, default=0.0)

        parser.add_argument("--young-modulus",      "-ym",  type=float, default=5e3)
        parser.add_argument("--poisson-ratio",      "-nu",  type=float, default=0.25)
        parser.add_argument("--friction",           "-mu",  type=float, default=0.8)
        parser.add_argument("--damping",                    type=float, default=3000.0)

        parser.add_argument("--yield-pressure",     "-yp",  type=float, default=50.0)
        parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=0.0)
        parser.add_argument("--yield-stress",       "-ys",  type=float, default=25.0)
        parser.add_argument("--hardening",                  type=float, default=0.5)

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
