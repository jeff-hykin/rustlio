#!/usr/bin/env python
"""Probe the Go2 onboard source (lidar/odom) vs fastlio in a recording.
Run in dimos env: cd ~/repos/dimos3 && uv run --no-sync python <this> --db <db>"""
import argparse
import numpy as np

from dimos.memory2.store.sqlite import SqliteStore
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.Odometry import Odometry


def traj(store, name, cls):
    rows = store.stream(name, cls).to_list()
    out = []
    for o in rows:
        if o.pose is None:
            continue
        t = o.pose_tuple  # (x,y,z,qx,qy,qz,qw)
        out.append((o.ts, float(t[0]), float(t[1]), float(t[2])))
    return np.array(out, dtype=np.float64)


def path_len(xyz):
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    args = ap.parse_args()
    store = SqliteStore(path=args.db)
    with store:
        fl = traj(store, "fastlio_odometry", Odometry)
        go = traj(store, "odom", PoseStamped)
        print(f"fastlio_odometry: n={len(fl)} extent x[{fl[:,1].min():.1f},{fl[:,1].max():.1f}] "
              f"y[{fl[:,2].min():.1f},{fl[:,2].max():.1f}] z[{fl[:,3].min():.1f},{fl[:,3].max():.1f}] "
              f"path={path_len(fl[:,1:]):.1f}m")
        print(f"odom(go2):        n={len(go)} extent x[{go[:,1].min():.1f},{go[:,1].max():.1f}] "
              f"y[{go[:,2].min():.1f},{go[:,2].max():.1f}] z[{go[:,3].min():.1f},{go[:,3].max():.1f}] "
              f"path={path_len(go[:,1:]):.1f}m")
        # start/end gap (loop closure check: does each return home?)
        print(f"fastlio start->end gap: {np.linalg.norm(fl[0,1:]-fl[-1,1:]):.2f}m")
        print(f"go2     start->end gap: {np.linalg.norm(go[0,1:]-go[-1,1:]):.2f}m")
        # one lidar cloud each: body vs world? centroid distance from sensor pose
        lc = store.stream("lidar", PointCloud2).to_list()[len(go)//2 if False else 100]
        flc = store.stream("fastlio_lidar", PointCloud2).to_list()[100]
        for tag, o in [("go2 lidar", lc), ("fastlio_lidar", flc)]:
            pts = np.asarray(o.data.pointcloud.points, dtype=np.float64)
            c = pts.mean(axis=0) if len(pts) else np.zeros(3)
            print(f"{tag}: pts={len(pts)} centroid=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) "
                  f"range[min..max dist]=[{np.linalg.norm(pts,axis=1).min():.1f}..{np.linalg.norm(pts,axis=1).max():.1f}]")


if __name__ == "__main__":
    main()
