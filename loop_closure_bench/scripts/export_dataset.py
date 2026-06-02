#!/usr/bin/env python
"""Export a dimos hk_village*.db recording to neutral files for the PGO loop-closure benchmark.

Must run inside the dimos environment (it uses dimos to decode the LCM/JPEG blobs and to
run the AprilTag detector). Example:

    cd ~/repos/dimos3
    uv run --no-sync python /Users/jeffhykin/repos/FASTLIO2_ROS2/loop_closure_bench/scripts/export_dataset.py \
        --db data/hk_village3.db --out /Users/jeffhykin/repos/FASTLIO2_ROS2/data/loop_bench/hk_village3

Outputs (all in --out):
  meta.json          camera intrinsics/extrinsics, counts, marker length
  lidar_poses.tum    one line per lidar frame: ts tx ty tz qx qy qz qw  (lidar-in-world, clean FAST-LIO)
  clouds.bin         packed body-frame clouds, frame order matches lidar_poses.tum.
                     per frame: int32 num_points, then num_points * 4 float32 (x,y,z,intensity)
  markers.json       list of raw AprilTag observations:
                     {ts, marker_id, t_cam_marker:[x,y,z,qx,qy,qz,qw] (marker in camera_optical, drift-free),
                      cam_in_world:[x,y,z,qx,qy,qz,qw] (camera_optical in world, from the clean recording),
                      lidar_idx (nearest lidar frame index by ts)}

Everything downstream (drift injection, groundtruth loop closures, C++ PGO harness, .rrd) reads
only these neutral files -- no dimos / AprilTag dependency.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dimos.memory2.store.sqlite import SqliteStore
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

# --- Go2 front camera (720p) intrinsics, from dimos _camera_info_static() ---
CAM_W, CAM_H = 1280, 720
CAM_K = np.array(
    [
        [797.47561649, 0.0, 643.53521678],
        [0.0, 796.48721128, 349.27836053],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
CAM_D = np.array([-0.07309429, -0.02341141, -0.00693059, 0.00923868], dtype=np.float64)

# Static mount chain base_link -> camera_link -> camera_optical, composed.
# (0.3,0,0) identity, then (0,0,0) rot (-0.5,0.5,-0.5,0.5). Result below.
BASE_TO_OPTICAL_7 = [0.3, 0.0, 0.0, -0.5, 0.5, -0.5, 0.5]


def _aruco_object_points(marker_length_m: float) -> np.ndarray:
    h = marker_length_m / 2.0
    return np.array(
        [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float32
    )


def estimate_marker_pose(corners_px, marker_length_m, cam_mtx, dist):
    obj = _aruco_object_points(marker_length_m)
    img = corners_px.reshape(4, 1, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam_mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return rvec, tvec


def create_aruco_detector(dictionary_name: str) -> "cv2.aruco.ArucoDetector":
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def _pose_to_tum_row(pose) -> tuple[float, float, float, float, float, float, float]:
    """dimos Observation.pose is a Pose (.position .orientation) on main."""
    q = pose.orientation
    return (
        float(pose.x), float(pose.y), float(pose.z),
        float(q.x), float(q.y), float(q.z), float(q.w),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--marker-length", type=float, default=0.10)
    ap.add_argument("--dictionary", default="DICT_APRILTAG_36h11")
    ap.add_argument("--lidar-stream", default="lidar", help="body-frame point cloud stream")
    ap.add_argument("--image-stream", default="color_image")
    ap.add_argument(
        "--skip-markers",
        action="store_true",
        help="only re-export clouds.bin + lidar_poses.tum (reuse existing markers.json/groundtruth)",
    )
    ap.add_argument(
        "--pose-stream",
        default="",
        help="stream carrying the FAST-LIO world pose (e.g. fastlio_odometry). "
        "If empty, each cloud's own .pose is used (hk_village). Otherwise the "
        "nearest-ts pose from this stream is assigned to each cloud.",
    )
    ap.add_argument(
        "--pose-from-payload",
        action="store_true",
        help="decode the pose-stream PoseStamped payload (o.data.x/y/z + "
        "orientation) instead of reading the indexed pose_* columns. Needed for "
        "the Go2 onboard `odom` stream, whose columns are zero placeholders.",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cam_mtx, dist = CAM_K, CAM_D
    detector = create_aruco_detector(args.dictionary)

    store = SqliteStore(path=args.db)
    with store:
        lidar = store.stream(args.lidar_stream, PointCloud2).to_list()
        images = store.stream(args.image_stream, Image).to_list()

        lidar_ts = np.array([o.ts for o in lidar], dtype=np.float64)

        # Resolve per-cloud world pose. Some recordings store the FAST-LIO pose
        # on the cloud row (hk_village); others leave it zero and carry the
        # trajectory in a separate higher-rate odometry stream matched by ts.
        pose_for_cloud = None
        if args.pose_stream and args.pose_from_payload:
            # Go2 `odom`: the pose is in the PoseStamped payload, not the indexed
            # pose_* columns (which are zero). Decode the payload via dimos.
            pstream = store.stream(args.pose_stream, PoseStamped).to_list()
            pts = np.array([o.ts for o in pstream], dtype=np.float64)
            pposes = np.array(
                [
                    [o.data.x, o.data.y, o.data.z,
                     o.data.orientation.x, o.data.orientation.y,
                     o.data.orientation.z, o.data.orientation.w]
                    for o in pstream
                ],
                dtype=np.float64,
            )

            def pose_for_cloud(ts: float):  # noqa: F811
                j = int(np.searchsorted(pts, ts))
                j = min(max(j, 0), len(pts) - 1)
                if j > 0 and abs(pts[j - 1] - ts) < abs(pts[j] - ts):
                    j -= 1
                return tuple(pposes[j])

        elif args.pose_stream:
            # The world pose lives on the pose-stream rows (pose_* columns), not
            # on the cloud rows. Read those columns directly via SQL -- avoids
            # decoding the (Odometry) payload and is far faster than iterating.
            import sqlite3

            conn = sqlite3.connect(args.db)
            rows = conn.execute(
                f"SELECT ts, pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw "
                f'FROM "{args.pose_stream}" WHERE pose_qw IS NOT NULL ORDER BY ts'
            ).fetchall()
            conn.close()
            arr = np.array(rows, dtype=np.float64)
            pts, pposes = arr[:, 0], arr[:, 1:8]

            def pose_for_cloud(ts: float):  # noqa: F811
                j = int(np.searchsorted(pts, ts))
                j = min(max(j, 0), len(pts) - 1)
                if j > 0 and abs(pts[j - 1] - ts) < abs(pts[j] - ts):
                    j -= 1
                return tuple(pposes[j])

        # --- lidar poses + clouds ---
        # The dimos cloud streams are stored in WORLD frame. The PGO (and the C++
        # harness) want BODY-frame clouds that it re-projects via each keyframe's
        # pose, so we "unregister" here: body = inv(pose) * world, using the same
        # pose written to lidar_poses.tum (round-trips exactly at zero drift).
        tum_lines = []
        with open(out / "clouds.bin", "wb") as cf:
            for o in lidar:
                if pose_for_cloud is not None:
                    x, y, z, qx, qy, qz, qw = pose_for_cloud(o.ts)
                else:
                    x, y, z, qx, qy, qz, qw = _pose_to_tum_row(o.pose)
                tum_lines.append(f"{o.ts:.9f} {x:.9f} {y:.9f} {z:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
                world = np.asarray(o.data.pointcloud.points, dtype=np.float64)
                n = world.shape[0]
                xyzi = np.zeros((n, 4), dtype=np.float32)
                if n:
                    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                    body = (world - np.array([x, y, z])) @ R  # R^T (world - t)
                    xyzi[:, :3] = body.astype(np.float32)
                cf.write(struct.pack("<i", n))
                cf.write(xyzi.tobytes())
        (out / "lidar_poses.tum").write_text("\n".join(tum_lines) + "\n")

        if args.skip_markers:
            print(f"[export] {args.db}: {len(lidar)} clouds re-exported (body frame), markers skipped -> {out}")
            return

        # --- marker detections (raw, drift-free camera<-marker + clean camera-in-world) ---
        markers = []
        for o in images:
            if o.pose is None:
                continue
            img = o.data
            bgr = img.numpy() if hasattr(img, "numpy") else np.asarray(img.data)
            corners, ids, _ = detector.detectMarkers(bgr)
            if ids is None:
                continue
            for c, mid in zip(corners, ids.flatten()):
                res = estimate_marker_pose(c, args.marker_length, cam_mtx, dist)
                if res is None:
                    continue
                rvec, tvec = res
                rot = Rotation.from_rotvec(rvec.reshape(3))
                q = rot.as_quat()  # x,y,z,w
                t = tvec.reshape(3)
                li = int(np.argmin(np.abs(lidar_ts - o.ts)))
                markers.append(
                    {
                        "ts": float(o.ts),
                        "marker_id": int(mid),
                        "t_cam_marker": [float(t[0]), float(t[1]), float(t[2]),
                                          float(q[0]), float(q[1]), float(q[2]), float(q[3])],
                        "cam_in_world": list(_pose_to_tum_row(o.pose)),
                        "lidar_idx": li,
                    }
                )
        (out / "markers.json").write_text(json.dumps(markers, indent=0))

        meta = {
            "db": args.db,
            "lidar_stream": args.lidar_stream,
            "pose_stream": args.pose_stream or args.lidar_stream,
            "image_stream": args.image_stream,
            "num_lidar": len(lidar),
            "num_images": len(images),
            "num_marker_obs": len(markers),
            "marker_length_m": args.marker_length,
            "dictionary": args.dictionary,
            "camera": {
                "width": CAM_W,
                "height": CAM_H,
                "K": [float(v) for v in CAM_K.reshape(-1)],
                "D": [float(v) for v in CAM_D.reshape(-1)],
            },
            "base_to_optical": BASE_TO_OPTICAL_7,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[export] {args.db}: {len(lidar)} lidar frames, {len(images)} images, {len(markers)} marker obs -> {out}")


if __name__ == "__main__":
    main()
