#!/usr/bin/env python3
"""
Extract the upstream C++ FAST-LIO trajectory (the `fastlio_odometry` stream) from
a DimOS memory sqlite DB into an npy of [t_epoch, x, y, z]. This is the reference
("hku-mars FAST-LIO") trajectory the Rust reimplementation is compared against.
"""
import sys, sqlite3
import numpy as np


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: extract_upstream.py <db> <out.npy> [stream=fastlio_odometry]")
    db, out = sys.argv[1], sys.argv[2]
    stream = sys.argv[3] if len(sys.argv) > 3 else "fastlio_odometry"
    con = sqlite3.connect(db)
    rows = con.execute(
        f"SELECT ts, pose_x, pose_y, pose_z FROM {stream} "
        f"WHERE pose_x IS NOT NULL ORDER BY ts"
    ).fetchall()
    if not rows:
        sys.exit(f"no rows with populated pose in stream '{stream}' of {db}")
    a = np.asarray(rows, float)
    np.save(out, a)
    t = a[:, 0]
    print(f"extracted {len(a)} {stream} poses -> {out}  ({t[-1]-t[0]:.1f}s span)")


if __name__ == "__main__":
    main()
