"""
Compute scooper (scoop_link) world-frame poses from recorded joint trajectories.

Reads:  dataset/demo_20260515_140725_dedup.npz   (states: N×6 joint angles)
Writes: dataset/demo_20260515_140725_scoop_poses.npz
          positions:   (N, 3) float32  [x, y, z] in world frame
          quaternions: (N, 4) float32  [qx, qy, qz, qw] in world frame

The URDF is placed with the same world transform used by simulation_newton_sand.py
so the poses are directly usable as kinematic targets in the refactored simulation.
"""

import numpy as np
import warp as wp
import newton

import argparse
import os

_URDF_PATH = "/home/gmr/Downloads/NewtonSimulation/ur_urdf/ur5_with_scoop.urdf"

def _parse_args():
    parser = argparse.ArgumentParser(description="Compute scooper world-frame poses from joint trajectory.")
    parser.add_argument("dataset", nargs="?", default="dataset/demo_20260515_140725_dedup.npz",
                        help="Path to .npz dataset file")
    parser.add_argument("--output", default=None,
                        help="Output path (default: same dir as input, _scoop_poses.npz suffix)")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Drop frames where step distance < this value (metres). "
                             "Removes pause segments where the scoop barely moves.")
    return parser.parse_args()

args = _parse_args()
_DATASET_PATH = args.dataset
_OUTPUT_PATH  = args.output or os.path.splitext(_DATASET_PATH)[0] + "_scoop_poses.npz"

_URDF_XFORM = wp.transform(
    wp.vec3(0.5, 0.0, 0.0),
    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
)

def main():
    # --- Build FK-only model (no MPM attributes or PD gains needed) ---
    builder = newton.ModelBuilder()
    builder.add_urdf(_URDF_PATH, xform=_URDF_XFORM, floating=False, enable_self_collisions=False)
    model = builder.finalize()

    scoop_idx = model.body_label.index("ur5_with_scoop/scoop_link")
    print(f"scoop_link body index: {scoop_idx}")

    # --- Load dataset ---
    data   = np.load(_DATASET_PATH)
    states = data["states"].astype(np.float32)  # (N, 6)
    n      = len(states)
    print(f"Dataset: {n} frames")

    # --- Allocate output ---
    positions   = np.zeros((n, 3), dtype=np.float32)
    quaternions = np.zeros((n, 4), dtype=np.float32)  # [qx, qy, qz, qw]

    # --- Compute FK for every frame ---
    state = model.state()
    jq    = state.joint_q.numpy().copy()  # shape (6,)

    for i, q in enumerate(states):
        jq[:6] = q
        state.joint_q.assign(jq)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        tf = state.body_q.numpy()[scoop_idx]  # [px, py, pz, qx, qy, qz, qw]
        positions[i]   = tf[:3]
        quaternions[i] = tf[3:]
        if (i + 1) % 200 == 0 or i == n - 1:
            print(f"  frame {i+1}/{n}  pos={positions[i]}")

    # --- Filter out pause frames ---
    if args.min_delta > 0.0:
        deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        keep = np.concatenate([[True], deltas >= args.min_delta])
        positions   = positions[keep]
        quaternions = quaternions[keep]
        print(f"\nAfter filtering (min_delta={args.min_delta}): {keep.sum()} / {n} frames kept")

    # --- Save ---
    np.savez(_OUTPUT_PATH, positions=positions, quaternions=quaternions)
    print(f"\nSaved to {_OUTPUT_PATH}")
    print(f"Position range:  min={positions.min(axis=0)}, max={positions.max(axis=0)}")
    print(f"Quaternion range: min={quaternions.min(axis=0)}, max={quaternions.max(axis=0)}")

if __name__ == "__main__":
    main()
