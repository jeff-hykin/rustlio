#!/usr/bin/env bash
# Quick ICP-experiment evaluator: runs the key loss + no-regress cases and prints
# a compact table. Pass extra pgo key=val args (e.g. loop_symmetric=1); set env
# (e.g. ICP_SYM=1) before calling. Baselines (committed e6d6ab5) in the labels.
#   ./icp_eval.sh [extra args...]
set -uo pipefail
cd "$(dirname "$0")"
EXTRA="$*"
KITTI="key_pose_delta_trans=2.0 loop_time_thresh=30 loop_search_radius=15 max_icp_correspondence_dist=5.0 loop_score_thresh=2.0 loop_submap_half_range=15 submap_resolution=0.5 max_loop_offset=8.0 loop_trans_floor=256"
FAST="key_pose_delta_trans=2.0 loop_time_tresh=25 loop_search_radius=10 max_icp_correspondence_dist=2.0 loop_score_tresh=2.0 loop_submap_half_range=10 submap_resolution=0.3 backend=rust loop_trans_scale=0.02"
IND="backend=rust loop_time_thresh=25 loop_search_radius=2.0 max_icp_correspondence_dist=1.0 loop_score_tresh=0.3 loop_submap_half_range=10 loop_source_submap_half_range=0 submap_resolution=0.2 max_loop_offset=2.0"
GO2="key_pose_delta_trans=1.0 loop_search_radius=20 loop_time_tresh=30 max_icp_correspondence_dist=2.0 loop_submap_half_range=15 submap_resolution=0.2 loop_score_tresh=2.0 backend=rust loop_trans_scale=0.02 use_scan_context=1 sc_max_range=8"
pgo(){ uv run -q --with numpy --with scipy python run_bench.py --in ../../data/loop_bench/$1 --yaw-per-m $2 "${@:3}" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(round(d['traj_ate_m']['pgo'],3))"; }
go2(){ uv run -q --with numpy python go2_eval.py --go2 ../../data/loop_bench/$1 --gt $2 "${@:3}" 2>/dev/null | python3 -c "import sys;[print(l.split('->')[1].split('m')[0].strip()) for l in sys.stdin if 'sim' in l]"; }
echo "kitti05_clean  (C++0.60 rust0.034): $(python3 ate.py 05 $KITTI $EXTRA | sed 's/.*ate_pgo=//')"
echo "stair_fl_clean (C++0.08 rust0.58 ): $(pgo stair_plaza 0.0 $FAST $EXTRA)"
echo "grass_fl_clean (C++1.26 rust3.61 ): $(pgo grass_field_loop 0.0 $FAST $EXTRA)"
echo "indoor_hk4_y0  (C++~0.6 rust0.72 ): $(pgo hk_village4 0.0 $IND $EXTRA)"
echo "go2_outdoor    (C++2.58 rust6.09 ): $(go2 outdoor_small_loop_go2 ~/datasets/fastlio_recordings/gtsam_odom.tum $GO2 $EXTRA)"
echo "go2_stair      (C++corr rust2.25 ): $(go2 stair_plaza_go2 ~/datasets/go2_recordings/2026-06-01_6-05pm-PST/gtsam_odom.tum $GO2 $EXTRA)"
