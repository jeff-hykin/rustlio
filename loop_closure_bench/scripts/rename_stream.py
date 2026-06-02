#!/usr/bin/env python
"""Rename a dimos memory2 PoseStamped stream by copying it to a new name and
deleting the old (avoids re-running detection). Run in the dimos env:

    cd ~/repos/dimos3 && uv run --no-sync python rename_stream.py --db <db> --old apriltags --new april_tags
"""
import argparse
from dimos.memory2.store.sqlite import SqliteStore
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    args = ap.parse_args()
    store = SqliteStore(path=args.db)
    with store:
        if args.old not in store.list_streams():
            print(f"no stream '{args.old}' in {args.db}; nothing to do")
            return
        if args.new in store.list_streams():
            store.delete_stream(args.new)
        src = store.stream(args.old, PoseStamped).to_list()
        dst = store.stream(args.new, PoseStamped)
        for o in src:
            dst.append(o.data, ts=o.ts, pose=o.pose_tuple, tags=o.tags)
        store.delete_stream(args.old)
    print(f"renamed '{args.old}' -> '{args.new}' ({len(src)} obs) in {args.db}")


if __name__ == "__main__":
    main()
