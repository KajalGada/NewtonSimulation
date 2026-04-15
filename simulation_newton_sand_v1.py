# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Newton MPM simulation of kinetic sand with a UR5e robot.
#
# Structure follows newton/examples/mpm/example_mpm_granular.py.
# Robot loading follows newton/examples/robot/example_robot_panda_hydro.py.
#
# Material parameters are tuned from granular_object.py to reproduce
# kinetic-sand behaviour: deformable, high-friction, heavily-damped,
# low yield-strength so the bulk flows under disturbance but holds its
# shape at rest.
#
# Robot control: kinematic FK drive.  Joint angles are interpolated
# between waypoints each frame; newton.eval_fk propagates those angles
# to body_q so that the MPM solver treats every robot link as a moving
# rigid collider automatically.

import sys

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.solvers import SolverImplicitMPM

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

# Hover above the sand pile (EE ≈ (−0.005, 0.000, 0.301))
_Q_ABOVE = np.array([-0.2696, -1.2040,  1.6280, -0.1908,  0.1230,  0.9396],
                    dtype=np.float32)

# Wrist at y = −0.13, z = 0.05 — entry point of sweep (EE ≈ (−0.003, −0.131, 0.049))
_Q_ENTER = np.array([-0.0066, -0.5588,  1.4786,  0.3709, -0.0002, -0.0004],
                    dtype=np.float32)

# Wrist at y =  0.00, z = 0.05 — midpoint of sweep  (EE ≈ (−0.003, 0.000, 0.049))
_Q_MID   = np.array([-0.2692, -0.5747,  1.5342,  0.3666,  0.0050,  0.0011],
                    dtype=np.float32)

# Wrist at y = +0.13, z = 0.05 — exit point of sweep (EE ≈ (−0.003, 0.130, 0.049))
_Q_EXIT  = np.array([-0.5143, -0.5587,  1.4782,  0.3708,  0.0049, -0.0002],
                    dtype=np.float32)

# Waypoint list: (target_joint_angles, duration_in_seconds)
# Sequence: hover → enter pile → sweep across → exit pile → hover (loops)
_WAYPOINTS = [
    (_Q_ABOVE, 2.5),   # settle above pile while sand falls under gravity
    (_Q_ENTER, 2.0),   # descend and enter pile at y = -0.13
    (_Q_MID,   2.5),   # sweep through first half of pile
    (_Q_EXIT,  2.5),   # sweep through second half of pile
    (_Q_ABOVE, 2.0),   # lift back out — loop repeats
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

        # ---- UR5e robot ------------------------------------------------
        # Placed at (0.5, 0, 0) with identity orientation.
        # shoulder_pan = π (set in initial joint_q below) rotates the arm
        # to face the sand pile at origin.
        ur5e_asset = newton.utils.download_asset("universal_robots_ur5e")
        builder.add_usd(
            str(ur5e_asset / "usd_structured/ur5e.usda"),
            xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
            enable_self_collisions=False,
        )

        # UR5e contributes 6 revolute DOFs (1 fixed base joint has 0 DOFs).
        # They are the first (and only) entries in builder.joint_q.
        self.ur5e_dof = 6
        builder.joint_q[:self.ur5e_dof] = _Q_ABOVE.tolist()

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

        # Use finite-difference body velocity: computes (body_q - body_q_prev) / dt
        # rather than reading body_qd, which is zero for our FK-driven kinematic robot.
        # Without this, the MPM solver treats the robot as stationary and particles
        # get no smooth push-away velocity from moving links.
        mpm_options.collider_velocity_mode = "finite_difference"

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.solver = SolverImplicitMPM(self.model, mpm_options)

        # Propagate home joint angles → body_q so the robot starts in the
        # correct pose rather than the default zero-angle configuration.
        self._apply_fk()

        # Re-register colliders now that body_q reflects the FK home pose.
        # This also initialises body_q_prev correctly (from state_0.body_q)
        # so the first substep sees zero body velocity instead of a huge spike
        # from zero-config → _Q_ABOVE.  Pass body_mass=0 so all robot links
        # are treated as kinematic (they push sand, but sand cannot push them).
        self.solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )

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
    # Robot FK helpers
    # ------------------------------------------------------------------
    def _set_joint_q(self, q: np.ndarray):
        """Write the 6 UR5e DOFs into model.joint_q."""
        q_np = self.model.joint_q.numpy()
        q_np[:self.ur5e_dof] = q
        self.model.joint_q.assign(q_np)

    def _apply_fk(self):
        """Propagate current model.joint_q → body_q in both states."""
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

    def _advance_robot_substep(self):
        """Interpolate joint angles along the waypoint sequence, advancing by sim_dt.

        Called once per substep so that body_q changes by exactly one substep's
        worth of motion each call.  The MPM solver computes body velocity as
        (body_q - body_q_prev) / sim_dt, so advancing by sim_dt here gives the
        correct physical velocity instead of a 4× spike from frame-level updates.
        """
        wp_target, wp_duration = _WAYPOINTS[self._wp_idx]
        prev_idx = (self._wp_idx - 1) % len(_WAYPOINTS)
        wp_prev, _ = _WAYPOINTS[prev_idx]

        t = min(self._wp_elapsed / wp_duration, 1.0)
        q_interp = (1.0 - t) * wp_prev + t * wp_target

        self._set_joint_q(q_interp)
        self._apply_fk()

        self._wp_elapsed += self.sim_dt  # advance by substep dt, not frame dt
        if self._wp_elapsed >= wp_duration:
            self._wp_idx = (self._wp_idx + 1) % len(_WAYPOINTS)
            self._wp_elapsed = 0.0

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    def capture(self):
        """CUDA graph capture when conditions allow."""
        self.graph = None
        if wp.get_device().is_cuda and self.solver.grid_type == "fixed":
            if self.sim_substeps % 2 != 0:
                wp.utils.warn("Sim substeps must be even for graph capture of MPM step")
            else:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            # Update robot pose once per substep so the velocity seen by the
            # MPM solver matches the actual arm speed (avoids the 4× spike that
            # occurs when body_q is updated only once per frame).
            self._advance_robot_substep()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver._project_outside(self.state_1, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

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
        parser.set_defaults(num_frames=700)

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
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.06)
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

        parser.add_argument("--young-modulus",      "-ym",  type=float, default=1.5e5)
        parser.add_argument("--poisson-ratio",      "-nu",  type=float, default=0.25)
        parser.add_argument("--friction",           "-mu",  type=float, default=1.2)
        parser.add_argument("--damping",                    type=float, default=800.0)

        parser.add_argument("--yield-pressure",     "-yp",  type=float, default=300.0)
        parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=0.0)
        parser.add_argument("--yield-stress",       "-ys",  type=float, default=150.0)
        parser.add_argument("--hardening",                  type=float, default=5.0)

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
