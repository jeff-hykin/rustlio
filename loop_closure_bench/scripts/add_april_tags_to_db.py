#!/usr/bin/env python
"""Detect AprilTags in a mem2.db's color_image stream and write them back as an
`april_tags` PoseStamped stream of RELATIVE poses (tag-in-camera, from solvePnP),
with marker_id in each observation's tags. Also dumps markers.json + meta.json for
the gtsam_odom solve. Run in the dimos env (needs cv2 + dimos).

    cd ~/repos/dimos3 && uv run --no-sync python add_april_tags_to_db.py \
        --db <mem2.db> --markers-out <markers.json> --meta-out <meta.json>
"""
import argparse, json
import numpy as np
from scipy.spatial.transform import Rotation

from dimos.memory2.store.sqlite import SqliteStore
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

# reuse the detection helpers + Go2 camera/extrinsic constants from export_dataset
from export_dataset import (
    create_aruco_detector, estimate_marker_pose, CAM_K, CAM_D, BASE_TO_OPTICAL_7,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--markers-out", required=True)
    ap.add_argument("--meta-out", required=True)
    ap.add_argument("--image-stream", default="color_image")
    ap.add_argument("--stream-name", default="april_tags")
    ap.add_argument("--marker-length", type=float, default=0.10)
    ap.add_argument("--dictionary", default="DICT_APRILTAG_36h11")
    args = ap.parse_args()

    detector = create_aruco_detector(args.dictionary)
    markers = []
    store = SqliteStore(path=args.db)
    with store:
        images = store.stream(args.image_stream, Image).to_list()
        if args.stream_name in store.list_streams():
            store.delete_stream(args.stream_name)
        aps = store.stream(args.stream_name, PoseStamped)
        for o in images:
            img = o.data
            bgr = img.numpy() if hasattr(img, "numpy") else np.asarray(img.data)
            corners, ids, _ = detector.detectMarkers(bgr)
            if ids is None:
                continue
            for c, mid in zip(corners, ids.flatten()):
                res = estimate_marker_pose(c, args.marker_length, CAM_K, CAM_D)
                if res is None:
                    continue
                rvec, tvec = res
                q = Rotation.from_rotvec(rvec.reshape(3)).as_quat()  # x,y,z,w
                t = tvec.reshape(3)
                tcm = [float(t[0]), float(t[1]), float(t[2]),
                       float(q[0]), float(q[1]), float(q[2]), float(q[3])]
                markers.append({"ts": float(o.ts), "marker_id": int(mid), "t_cam_marker": tcm})
                ps = PoseStamped(ts=float(o.ts), position=tcm[:3], orientation=tcm[3:])
                aps.append(ps, ts=float(o.ts), pose=tuple(tcm), tags={"marker_id": int(mid)})
    json.dump(markers, open(args.markers_out, "w"))
    json.dump({"base_to_optical": BASE_TO_OPTICAL_7}, open(args.meta_out, "w"))
    ids = sorted({m["marker_id"] for m in markers})
    print(f"[april_tags] {len(markers)} detections, markers {ids} -> stream '{args.stream_name}' + {args.markers_out}")


if __name__ == "__main__":
    main()
