#!/usr/bin/env python
"""Build a drift-corrected groundtruth trajectory (gtsam_odom) by landmark SLAM:
fastlio odometry as a locally-correct pose chain + AprilTag observations as static
landmark constraints, solved globally with GTSAM.

fastlio is locally accurate but drifts globally (e.g. ~130 m on the grass field).
A static AprilTag seen at different times must be in ONE place, so its repeated
observations pin the chain and remove the accumulated drift -> a trajectory that
agrees with where the tags actually are. We trust the tag POSITION (solvePnP is
metric) but distrust its ORIENTATION (a small planar tag is yaw/pitch-ambiguous),
and wrap tag factors in a robust kernel so a bad detection can't dominate.

    uv run --with gtsam --with numpy python build_gtsam_gt.py \
        --db <mem2.db> --markers <markers.json> --meta <meta.json> --out <gtsam_odom.tum>

Run in: a python env with gtsam + numpy (uv --with gtsam --with numpy).
"""
import argparse, json, sqlite3
import numpy as np
import gtsam
from gtsam import Pose3, Rot3, Point3, BetweenFactorPose3, PriorFactorPose3
from gtsam.symbol_shorthand import X, L


def pose_from7(p):  # [x,y,z,qx,qy,qz,qw] -> gtsam.Pose3
    return Pose3(Rot3.Quaternion(p[6], p[3], p[4], p[5]), Point3(p[0], p[1], p[2]))


def pose_to7(P):
    q = P.rotation().toQuaternion(); t = P.translation()
    return [t[0], t[1], t[2], q.x(), q.y(), q.z(), q.w()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--markers", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pose-stream", default="fastlio_odometry")
    ap.add_argument("--node-stride", type=int, default=3, help="subsample fastlio nodes")
    ap.add_argument("--odom-rot-sig", type=float, default=0.004)   # rad/edge (fastlio local)
    ap.add_argument("--odom-trans-sig", type=float, default=0.02)  # m/edge
    ap.add_argument("--tag-rot-sig", type=float, default=1.0)      # rad (distrust tag orientation)
    ap.add_argument("--tag-trans-sig", type=float, default=0.1)    # m (trust tag position)
    ap.add_argument("--tag-huber", type=float, default=0.5)        # robust kernel (m, whitened)
    args = ap.parse_args()

    # --- fastlio odometry nodes (pose columns) ---
    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        f'SELECT ts,pose_x,pose_y,pose_z,pose_qx,pose_qy,pose_qz,pose_qw '
        f'FROM "{args.pose_stream}" WHERE pose_qw IS NOT NULL ORDER BY ts'
    ).fetchall()
    conn.close()
    rows = rows[::args.node_stride]
    node_ts = np.array([r[0] for r in rows])
    node_pose = [pose_from7(r[1:8]) for r in rows]
    n = len(rows)
    print(f"nodes: {n} (stride {args.node_stride})")

    meta = json.load(open(args.meta))
    T_bo = pose_from7(meta["base_to_optical"])  # camera_optical -> base_link
    tags = json.load(open(args.markers))
    print(f"tag detections: {len(tags)}")

    def nearest_node(ts):
        j = int(np.searchsorted(node_ts, ts)); j = min(max(j, 0), n - 1)
        if j > 0 and abs(node_ts[j - 1] - ts) < abs(node_ts[j] - ts):
            j -= 1
        return j

    graph = gtsam.NonlinearFactorGraph()
    init = gtsam.Values()
    prior_n = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-4))
    odom_n = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([args.odom_rot_sig] * 3 + [args.odom_trans_sig] * 3)
    )
    base_tag_n = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([args.tag_rot_sig] * 3 + [args.tag_trans_sig] * 3)
    )
    tag_n = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(args.tag_huber), base_tag_n
    )

    for i in range(n):
        init.insert(X(i), node_pose[i])
    graph.add(PriorFactorPose3(X(0), node_pose[0], prior_n))
    for i in range(n - 1):
        rel = node_pose[i].between(node_pose[i + 1])
        graph.add(BetweenFactorPose3(X(i), X(i + 1), rel, odom_n))

    # tag landmark (Pose3) factors: L_m (world) = X_i (world) ∘ (T_bo ∘ M_cam_tag)
    seen = set()
    n_tagf = 0
    for d in tags:
        mid = int(d["marker_id"])
        i = nearest_node(d["ts"])
        T_body_tag = T_bo.compose(pose_from7(d["t_cam_marker"]))
        if mid not in seen:
            init.insert(L(mid), node_pose[i].compose(T_body_tag))
            seen.add(mid)
        graph.add(BetweenFactorPose3(X(i), L(mid), T_body_tag, tag_n))
        n_tagf += 1
    print(f"tag landmarks: {sorted(seen)} | tag factors: {n_tagf}")

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    opt = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
    print(f"initial error: {graph.error(init):.1f}")
    result = opt.optimize()
    print(f"final error:   {graph.error(result):.1f}  ({opt.iterations()} iters)")

    with open(args.out, "w") as f:
        for i in range(n):
            p = pose_to7(result.atPose3(X(i)))
            f.write(f"{node_ts[i]:.9f} " + " ".join(f"{v:.9f}" for v in p) + "\n")
    # report how much the trajectory moved (drift removed)
    moved = [np.linalg.norm(result.atPose3(X(i)).translation() - node_pose[i].translation()) for i in range(n)]
    print(f"wrote {args.out} | max correction {max(moved):.2f} m, mean {np.mean(moved):.2f} m")


if __name__ == "__main__":
    main()
