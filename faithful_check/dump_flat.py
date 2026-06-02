#!/usr/bin/env python3
"""
Dump a raw Livox Mid-360 pcap into a flat binary that both the Rust rustlio
flat-runner and the C++ reference harness read, so both LIO implementations
consume byte-identical IMU + lidar input (removing any parse/format ambiguity
from the comparison).

Parsing mirrors rustlio's parse_livox_custom_msg / parse_imu_cdr exactly:
  - accel converted g -> m/s^2 with *9.80665
  - lidar points kept where line<4 and (tag&0x30) in {0x10,0x00}, range-gated,
    every `filter_num`-th point, curvature = offset_time_ns/1e6 (ms), plus the
    consecutive-duplicate drop rustlio applies.

Flat format (little-endian):
  magic "FLT1" (4 bytes)
  u64 n_imu, u64 n_frames
  n_imu   x (f64 t, ax, ay, az, gx, gy, gz)         # accel already m/s^2
  n_frames x (f64 t_start, u64 npts, npts x (f32 x,y,z,intensity,curv_ms))
"""
import struct, argparse
from pcap_to_mcap import iter_packets, LIDAR_PORT, IMU_PORT, DT_IMU, DT_CART_HIGH, DT_CART_LOW, HDR

ACC_SCALE = 9.80665
MIN_R, MAX_R = 0.5, 20.0
FILTER_NUM = 3


def valid(tag, line):
    return line < 4 and ((tag & 0x30) == 0x10 or (tag & 0x30) == 0x00)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap")
    ap.add_argument("out_flat")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--frame-ms", type=float, default=100.0)
    args = ap.parse_args()

    clock_off = None
    t0 = None
    frame_ns = int(args.frame_ms * 1e6)
    cur_start = None
    cur_pts = []          # (curv_ms, x, y, z, intensity)
    prev_xyz = None
    imus = []             # (t, ax, ay, az, gx, gy, gz)
    frames = []           # (t_start_epoch, [pts])

    def flush():
        if cur_pts:
            frames.append(((cur_start + clock_off) / 1e9, list(cur_pts)))

    for pcap_ts, port, pl in iter_packets(args.pcap):
        if len(pl) < HDR:
            continue
        is_lidar = port == LIDAR_PORT and pl[10] in (DT_CART_HIGH, DT_CART_LOW)
        is_imu = port == IMU_PORT and pl[10] == DT_IMU
        if not (is_lidar or is_imu):
            continue
        sensor_ts = struct.unpack("<Q", pl[28:36])[0]
        if clock_off is None:
            clock_off = pcap_ts - sensor_ts
            t0 = sensor_ts
        if args.max_seconds and (sensor_ts - t0) > args.max_seconds * 1e9:
            break

        if is_lidar:
            dot_num = struct.unpack("<H", pl[5:7])[0]
            time_interval = struct.unpack("<H", pl[3:5])[0]
            if cur_start is None:
                cur_start = sensor_ts
            if sensor_ts - cur_start >= frame_ns:
                flush(); cur_pts.clear(); prev_xyz = None; cur_start = sensor_ts
            per_pt = (time_interval * 100) / dot_num if dot_num else 0.0
            pkt_off = sensor_ts - cur_start
            pts = pl[HDR:]
            if pl[10] == DT_CART_HIGH:
                psz, sc = 14, 1000.0
                rd = lambda o: struct.unpack("<iii", pts[o:o + 12])
            else:
                psz, sc = 8, 100.0
                rd = lambda o: struct.unpack("<hhh", pts[o:o + 6])
            i = 0
            while i < dot_num:
                o = i * psz
                if o + psz > len(pts):
                    break
                xi, yi, zi = rd(o)
                refl, tag = pts[o + psz - 2], pts[o + psz - 1]
                line = 0
                x, y, z = xi / sc, yi / sc, zi / sc
                r2 = x * x + y * y + z * z
                if valid(tag, line) and MIN_R * MIN_R <= r2 <= MAX_R * MAX_R:
                    dup = prev_xyz is not None and abs(x - prev_xyz[0]) <= 1e-7 \
                        and abs(y - prev_xyz[1]) <= 1e-7 and abs(z - prev_xyz[2]) <= 1e-7
                    if not dup:
                        curv = (pkt_off + i * per_pt) / 1e6  # ns -> ms
                        cur_pts.append((curv, x, y, z, float(refl)))
                        prev_xyz = (x, y, z)
                i += FILTER_NUM
        else:  # imu
            gx, gy, gz, ax, ay, az = struct.unpack("<6f", pl[HDR:HDR + 24])
            t = (sensor_ts + clock_off) / 1e9
            imus.append((t, ax * ACC_SCALE, ay * ACC_SCALE, az * ACC_SCALE, gx, gy, gz))
    flush()

    with open(args.out_flat, "wb") as f:
        f.write(b"FLT1")
        f.write(struct.pack("<QQ", len(imus), len(frames)))
        for rec in imus:
            f.write(struct.pack("<7d", *rec))
        for t_start, pts in frames:
            f.write(struct.pack("<dQ", t_start, len(pts)))
            for curv, x, y, z, inten in pts:
                f.write(struct.pack("<5f", x, y, z, inten, curv))
    npts = sum(len(p) for _, p in frames)
    print(f"wrote {args.out_flat}: {len(imus)} imu, {len(frames)} frames, {npts} pts")


if __name__ == "__main__":
    main()
