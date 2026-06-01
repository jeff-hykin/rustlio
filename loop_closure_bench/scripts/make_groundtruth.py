#!/usr/bin/env python
"""Derive groundtruth loop-closure events from AprilTag co-observations.

A static marker seen at two temporally-separated times means the robot revisited
that place -> a true loop closure. The relative camera transform between the two
observations is recovered purely from the two solvePnP marker poses, so it is
drift-FREE (independent of odometry). We save these events so downstream systems
can evaluate / constrain loop closure without running AprilTag detection.

    python make_groundtruth.py --in data/loop_bench/hk_village3

Writes <in>/groundtruth.json:
  marker_world   marker_id -> best-estimate true world position (clean-traj centroid)
  visits         marker_id -> list of temporal visit clusters {start,end,n,repr_idx,repr_ts}
  loop_events    list of {marker_id, idx_a, idx_b, ts_a, ts_b, dt,
                          rel_cam_a_in_b:[7]}  # drift-free relative pose, cam_a expressed in cam_b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench_lib import load_dataset, marker_world_positions, pose_from_row, pose_inv, pose_mul, pose_to_row

VISIT_GAP_S = 15.0   # split a marker's observations into visits on gaps larger than this
MIN_DT_S = 10.0      # a loop event must connect observations at least this far apart in time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--visit-gap", type=float, default=VISIT_GAP_S)
    ap.add_argument("--min-dt", type=float, default=MIN_DT_S)
    args = ap.parse_args()

    ds = load_dataset(args.inp)

    # group observations per marker, sorted by ts
    by_marker: dict[int, list[dict]] = {}
    for m in ds.markers:
        by_marker.setdefault(int(m["marker_id"]), []).append(m)
    for v in by_marker.values():
        v.sort(key=lambda m: m["ts"])

    # best-estimate true marker world position from the clean trajectory
    clean_pos = marker_world_positions(ds)
    marker_world = {str(k): clean_pos[k].mean(0).tolist() for k in clean_pos}

    visits: dict[str, list[dict]] = {}
    loop_events: list[dict] = []

    for mid, obs in by_marker.items():
        ts = np.array([o["ts"] for o in obs])
        # split into visits on temporal gaps
        splits = np.where(np.diff(ts) > args.visit_gap)[0] + 1
        groups = np.split(np.arange(len(obs)), splits)
        vlist = []
        for g in groups:
            # representative obs of a visit = the one where the marker is closest
            # to the camera (smallest tvec norm) -> most accurate solvePnP pose
            dists = [np.linalg.norm(np.asarray(obs[i]["t_cam_marker"])[:3]) for i in g]
            rep = int(g[int(np.argmin(dists))])
            vlist.append(
                {
                    "start_ts": float(ts[g[0]]),
                    "end_ts": float(ts[g[-1]]),
                    "n": int(len(g)),
                    "repr_idx": int(obs[rep]["lidar_idx"]),
                    "repr_ts": float(obs[rep]["ts"]),
                    "_rep_obs": obs[rep],
                }
            )
        visits[str(mid)] = [{k: v for k, v in d.items() if k != "_rep_obs"} for d in vlist]

        # loop events: every pair of visits to the same marker
        for i in range(len(vlist)):
            for j in range(i + 1, len(vlist)):
                a, b = vlist[i]["_rep_obs"], vlist[j]["_rep_obs"]
                if abs(a["ts"] - b["ts"]) < args.min_dt:
                    continue
                m_ca = pose_from_row(np.asarray(a["t_cam_marker"]))  # marker in cam_a
                m_cb = pose_from_row(np.asarray(b["t_cam_marker"]))  # marker in cam_b
                # cam_a expressed in cam_b frame = C_b^-1 ∘ C_a = M_cb ∘ M_ca^-1  (drift-free)
                rel = pose_mul(m_cb, pose_inv(m_ca))
                loop_events.append(
                    {
                        "marker_id": mid,
                        "idx_a": int(a["lidar_idx"]),
                        "idx_b": int(b["lidar_idx"]),
                        "ts_a": float(a["ts"]),
                        "ts_b": float(b["ts"]),
                        "dt": float(abs(a["ts"] - b["ts"])),
                        "rel_cam_a_in_b": pose_to_row(rel),
                    }
                )

    out = {
        "source": str(Path(args.inp).name),
        "marker_world": marker_world,
        "visits": visits,
        "loop_events": loop_events,
    }
    path = Path(args.inp) / "groundtruth.json"
    path.write_text(json.dumps(out, indent=2))
    nvisits = {k: len(v) for k, v in visits.items()}
    print(f"[groundtruth] markers={list(by_marker)} visits={nvisits} loop_events={len(loop_events)} -> {path}")


if __name__ == "__main__":
    main()
