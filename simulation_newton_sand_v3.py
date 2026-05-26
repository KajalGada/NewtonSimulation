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
# Joint angles are replayed from a recorded demonstration (.npz) at the
# dataset's native sample rate and interpolated to the simulation frame rate.
# SolverImplicitMPM handles particle↔rigid collision separately.

import sys

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM, SolverMuJoCo

_DATASET_PATH = "dataset/demo_20260515_140725_dedup.npz"


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
        # Load recorded demonstration
        # ------------------------------------------------------------------
        data = np.load(_DATASET_PATH)
        # states: (N, 6) float64 — joint positions in UR5 joint order
        self._dataset_states = data["states"].astype(np.float32)
        self._dataset_hz = args.dataset_hz
        self._dataset_len = len(self._dataset_states)

        q_start = self._dataset_states[0]

        # ------------------------------------------------------------------
        # Scene geometry
        # ------------------------------------------------------------------
        builder = newton.ModelBuilder()

        # Must be called before any bodies / particles are added
        SolverImplicitMPM.register_custom_attributes(builder)

        # ---- UR5 + scoop robot -----------------------------------------
        _URDF_PATH = "/home/gmr/Downloads/NewtonSimulation/ur_urdf/ur5_with_scoop.urdf"
        builder.add_urdf(
            _URDF_PATH,
            xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi)),
            floating=False,
            enable_self_collisions=False,
        )

        self.ur5e_dof = 6
        builder.joint_q[:self.ur5e_dof] = q_start.tolist()

        # PD gains
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 2000.0
            builder.joint_target_kd[i] =  100.0

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

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        # ---- Static container box --------------------------------------
        box_width      = 0.35
        box_depth      = 0.35
        box_height     = 0.05
        wall_thickness = 0.02
        box_y          = -0.2   # offset along Y

        box_cfg = newton.ModelBuilder.ShapeConfig(mu=0.6, gap=0.01)

        builder.add_shape_box(
            body=-1, cfg=box_cfg,
            xform=wp.transform(wp.vec3(0.0, box_y, wall_thickness * 0.5), wp.quat_identity()),
            hx=box_width * 0.5, hy=box_depth * 0.5, hz=wall_thickness * 0.5,
        )
        builder.add_shape_box(
            body=-1, cfg=box_cfg,
            xform=wp.transform(wp.vec3(0.0, box_y - box_depth * 0.5, box_height * 0.5), wp.quat_identity()),
            hx=box_width * 0.5, hy=wall_thickness * 0.5, hz=box_height * 0.5,
        )
        builder.add_shape_box(
            body=-1, cfg=box_cfg,
            xform=wp.transform(wp.vec3(0.0, box_y + box_depth * 0.5, box_height * 0.5), wp.quat_identity()),
            hx=box_width * 0.5, hy=wall_thickness * 0.5, hz=box_height * 0.5,
        )
        builder.add_shape_box(
            body=-1, cfg=box_cfg,
            xform=wp.transform(wp.vec3(-box_width * 0.5, box_y, box_height * 0.5), wp.quat_identity()),
            hx=wall_thickness * 0.5, hy=box_depth * 0.5, hz=box_height * 0.5,
        )
        builder.add_shape_box(
            body=-1, cfg=box_cfg,
            xform=wp.transform(wp.vec3(box_width * 0.5, box_y, box_height * 0.5), wp.quat_identity()),
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

        # finite_difference computes collider velocity as (body_q - body_q_prev) / dt.
        # With interleaved stepping (1 robot substep + 1 MPM substep, both at sim_dt),
        # the displacement and dt cancel correctly → exact scoop velocity, no N× overestimate.
        mpm_options.collider_velocity_mode = "finite_difference"

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )

        self.robot_solver = SolverMuJoCo(self.model)

        self.control = self.model.control()
        pos_np = self.control.joint_target_pos.numpy()
        pos_np[:self.ur5e_dof] = q_start
        self.control.joint_target_pos.assign(pos_np)

        # ------------------------------------------------------------------
        # Viewer setup
        # ------------------------------------------------------------------
        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(wp.vec3(1.5, -1.5, 1.2), pitch=-30.0, yaw=135.0)
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self.viewer.show_particles = True

    # ------------------------------------------------------------------
    # Robot control
    # ------------------------------------------------------------------
    def _update_robot_target(self):
        """Set PD target by interpolating the recorded dataset at sim_time."""
        t = self.sim_time * self._dataset_hz
        idx_lo = int(t)
        idx_hi = idx_lo + 1

        if idx_lo >= self._dataset_len - 1:
            q_target = self._dataset_states[-1]
        else:
            frac = t - idx_lo
            q_target = (1.0 - frac) * self._dataset_states[idx_lo] + frac * self._dataset_states[idx_hi]

        pos_np = self.control.joint_target_pos.numpy()
        pos_np[:self.ur5e_dof] = q_target
        self.control.joint_target_pos.assign(pos_np)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    def step(self):
        self._update_robot_target()

        # Interleave one robot substep with one MPM substep so that each MPM
        # step sees a body_q that advanced by exactly sim_dt.  With finite_difference
        # mode this means velocity = Δbody_q / sim_dt = exact scoop velocity (no N×
        # overestimate).  The smaller sim_dt per MPM step also cuts tunneling risk
        # by sim_substeps× vs running all MPM steps after a full-frame robot advance.
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.robot_solver.step(self.state_0, self.state_1, self.control, contacts=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            # Project any particles that tunnelled through the scoop back outside before
            # the grid-level solve runs.  body_q_prev still holds the pre-robot-step
            # position so the velocity estimate is correct.
            self.mpm_solver._project_outside(self.state_0, self.state_0, self.sim_dt)
            self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.sim_dt)

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

        # Dataset is 2774 frames at 125 Hz ≈ 22 s → ~1340 frames at 60 fps.
        parser.set_defaults(num_frames=1400)

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

        # Dataset playback rate (Hz) — must match how the .npz was recorded
        parser.add_argument("--dataset-hz", type=float, default=125.0,
                            help="Sample rate of the recorded demonstration (default: 125 Hz)")

        # Particle volume
        parser.add_argument("--emit-lo", type=float, nargs=3, default=[-0.15, -0.35, 0.02])
        parser.add_argument("--emit-hi", type=float, nargs=3, default=[ 0.15, -0.05, 0.20])
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
        )

        # Kinetic-sand material parameters
        parser.add_argument("--density",            type=float, default=1400.0)
        parser.add_argument("--air-drag",           type=float, default=6.0)
        parser.add_argument("--critical-fraction",  "-cf", type=float, default=0.0)

        parser.add_argument("--young-modulus",      "-ym",  type=float, default=8000)
        parser.add_argument("--poisson-ratio",      "-nu",  type=float, default=0.4)
        parser.add_argument("--friction",           "-mu",  type=float, default=1.4)
        parser.add_argument("--damping",                    type=float, default=400.0)

        parser.add_argument("--yield-pressure",     "-yp",  type=float, default=50.0)
        parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=0.2)
        parser.add_argument("--yield-stress",       "-ys",  type=float, default=100.0)
        parser.add_argument("--hardening",                  type=float, default=1.0)

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
