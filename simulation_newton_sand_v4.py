# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Newton MPM simulation of kinetic sand with a kinematic scoop end-effector.
#
# Single-solver variant: MuJoCo and the UR5 robot arm are removed entirely.
# The scoop is a standalone kinematic rigid body whose Cartesian pose is
# replayed from a pre-recorded dataset of world-frame scoop poses.
# SolverImplicitMPM handles all particle physics and particle↔scoop collision.

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

_SCOOP_STL_PATH = "ur_urdf/ur5_scoop.stl"
# _SCOOP_DATASET_PATH = "dataset/demo_20260515_140725_scoop_poses.npz"
# _SCOOP_DATASET_PATH = "dataset/demo_20260518_143230_scoop_poses.npz"
_SCOOP_DATASET_PATH = "dataset/demo_20260518_143752_scoop_poses.npz"


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
        # Load scoop pose dataset
        # ------------------------------------------------------------------
        _data = np.load(_SCOOP_DATASET_PATH)
        self._dataset_positions   = _data["positions"].astype(np.float32)   # (N, 3)
        self._dataset_quaternions = _data["quaternions"].astype(np.float32) # (N, 4) xyzw
        self._dataset_hz  = args.dataset_hz
        self._dataset_len = len(self._dataset_positions)

        # ------------------------------------------------------------------
        # Scene geometry
        # ------------------------------------------------------------------
        builder = newton.ModelBuilder()

        # Must be called before any bodies / particles are added
        SolverImplicitMPM.register_custom_attributes(builder)

        # ---- Kinematic scoop body --------------------------------------
        # Load STL via trimesh; mesh is stored in millimetres in the file.
        _scoop_raw = trimesh.load(_SCOOP_STL_PATH)
        _vertices  = np.array(_scoop_raw.vertices, dtype=np.float32) * 0.001
        _indices   = np.array(_scoop_raw.faces,    dtype=np.int32).flatten()
        scoop_mesh = newton.Mesh(_vertices, _indices)

        # Initial pose from the first dataset frame.
        _p0 = self._dataset_positions[0]
        _q0 = self._dataset_quaternions[0]
        initial_pos = wp.vec3(float(_p0[0]), float(_p0[1]), float(_p0[2]))
        initial_rot = wp.quat(float(_q0[0]), float(_q0[1]), float(_q0[2]), float(_q0[3]))

        _particle_radius = args.voxel_size / args.particles_per_cell * 0.5

        self.scoop_body_idx = builder.add_body(
            xform=wp.transform(initial_pos, initial_rot),
            mass=0.5,
            is_kinematic=True,
            label="scoop",
        )
        builder.add_shape_mesh(
            body=self.scoop_body_idx,
            mesh=scoop_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.6, density=0.0),
        )
        # Push particles to this distance outside the scoop surface to avoid
        # zero-separation meshing at the bowl floor (same logic as robot arm
        # version).
        builder.shape_margin[-1] = _particle_radius * 0.5

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
        # With per-substep body_q updates (advancing time by sim_dt each iteration)
        # the displacement and dt cancel correctly → exact scoop velocity.
        mpm_options.collider_velocity_mode = "finite_difference"

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()  # unused but kept for API symmetry

        # Set scoop to its initial Cartesian pose.  body_q layout: [px,py,pz, qx,qy,qz,qw].
        bq = self.state_0.body_q.numpy()
        bq[self.scoop_body_idx] = [*self._dataset_positions[0], *self._dataset_quaternions[0]]
        self.state_0.body_q.assign(bq)

        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )

        # ------------------------------------------------------------------
        # Viewer setup
        # ------------------------------------------------------------------
        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(wp.vec3(1.5, -1.5, 1.2), pitch=-30.0, yaw=135.0)
            self.viewer.camera.fov = 15.0
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self.viewer.show_particles = True

    # ------------------------------------------------------------------
    # Scoop trajectory
    # ------------------------------------------------------------------
    def _get_scoop_pose(self, t: float):
        """Return (pos, quat) by linearly interpolating the dataset at time t."""
        t_idx = t * self._dataset_hz
        idx_lo = int(t_idx)
        idx_hi = idx_lo + 1
        if idx_lo >= self._dataset_len - 1:
            return self._dataset_positions[-1], self._dataset_quaternions[-1]
        frac = t_idx - idx_lo
        pos  = (1.0 - frac) * self._dataset_positions[idx_lo]  + frac * self._dataset_positions[idx_hi]
        quat = (1.0 - frac) * self._dataset_quaternions[idx_lo] + frac * self._dataset_quaternions[idx_hi]
        quat = quat / np.linalg.norm(quat)
        return pos, quat

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    def step(self):
        # Update scoop pose each substep so finite_difference velocity mode
        # sees exactly sim_dt of displacement per MPM step.
        for i in range(self.sim_substeps):
            t = self.sim_time + i * self.sim_dt
            pos, quat = self._get_scoop_pose(t)

            bq = self.state_0.body_q.numpy()
            bq[self.scoop_body_idx] = [*pos, *quat]
            self.state_0.body_q.assign(bq)

            self.mpm_solver._project_outside(
                self.state_0, self.state_0, self.sim_dt,
                max_dist=self.mpm_solver.voxel_size * 0.5,
            )
            self.mpm_solver.step(
                self.state_0, self.state_0,
                contacts=None, control=None, dt=self.sim_dt,
            )

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

        # Dataset is 1383 frames at 125 Hz ≈ 11 s → ~660 frames at 60 fps.
        # Default to 700 so the scoop holds its final pose briefly before stopping.
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

        # Dataset playback rate (Hz) — must match how the scoop poses were recorded
        parser.add_argument("--dataset-hz", type=float, default=125.0,
                            help="Sample rate of the scoop pose dataset (default: 125 Hz)")

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
        parser.add_argument("--critical-fraction",  "-cf", type=float, default=0.15)

        parser.add_argument("--young-modulus",      "-ym",  type=float, default=8000)
        parser.add_argument("--poisson-ratio",      "-nu",  type=float, default=0.4)
        parser.add_argument("--friction",           "-mu",  type=float, default=1.4)
        parser.add_argument("--damping",                    type=float, default=450.0)

        parser.add_argument("--yield-pressure",     "-yp",  type=float, default=50.0)
        parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=0.0)
        parser.add_argument("--yield-stress",       "-ys",  type=float, default=100.0)
        parser.add_argument("--hardening",                  type=float, default=1.0)

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
