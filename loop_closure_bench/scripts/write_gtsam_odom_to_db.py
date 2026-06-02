#!/usr/bin/env python
"""Append a gtsam_odom.tum trajectory into a dimos memory2 mem2.db as a new
`gtsam_odom` PoseStamped stream (drift-corrected groundtruth from
build_gtsam_gt.py). Run in the dimos env:

    cd ~/repos/dimos3 && uv run --no-sync python <this> --db <mem2.db> --tum <gtsam_odom.tum>
"""
import argparse
from dimos.memory2.store.sqlite import SqliteStore
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--tum", required=True)
    ap.add_argument("--stream", default="gtsam_odom")
    args = ap.parse_args()

    rows = []
    for line in open(args.tum):
        v = [float(x) for x in line.split()]
        if len(v) >= 8:
            rows.append(v[:8])  # ts x y z qx qy qz qw

    store = SqliteStore(path=args.db)
    with store:
        if args.stream in store.list_streams():
            print(f"stream {args.stream} exists -> deleting and rewriting")
            store.delete_stream(args.stream)
        s = store.stream(args.stream, PoseStamped)
        for ts, x, y, z, qx, qy, qz, qw in rows:
            ps = PoseStamped(ts=ts, position=[x, y, z], orientation=[qx, qy, qz, qw])
            s.append(ps, ts=ts, pose=(x, y, z, qx, qy, qz, qw))
    print(f"wrote {len(rows)} poses to stream '{args.stream}' in {args.db}")


if __name__ == "__main__":
    main()
