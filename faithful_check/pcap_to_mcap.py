#!/usr/bin/env python3
"""
Convert a raw Livox Mid-360 pcap (Livox-SDK2 UDP packets) into an MCAP carrying
  /livox/lidar : livox_ros_driver2 CustomMsg   (CDR)
  /livox/imu   : sensor_msgs/Imu               (CDR)
whose byte layouts exactly match rustlio2's parse_livox_custom_msg /
parse_imu_cdr (rust/src/main.rs). This lets the *exact same raw sensor stream*
that upstream C++ FAST-LIO consumed be fed to the Rust reimplementation, for a
faithfulness comparison.

Why pcap and not the DimOS .db: the .db's `fastlio_lidar`/`lidar` clouds decode
to world-frame open3d tensors that have dropped per-point offset_time/tag/line —
useless as raw LIO input. The pcap is the ground-truth raw sensor data.

Clock: pcap capture timestamps are epoch; Livox `sensor_ts` is sensor-uptime ns.
We stamp every frame/sample with sensor_ts + (epoch - sensor_ts)@packet0, giving
an internally-consistent sensor clock that is also epoch-comparable to the db
`fastlio_odometry` reference. Lidar and IMU share the one offset.

Packet layout (Livox-SDK2 LivoxLidarEthernetPacket, validated against the
recording's decoded jsonl):
  off 0  : version u8
  off 1  : length u16, time_interval u16 (unit 0.1us), dot_num u16, udp_cnt u16
  off 9  : frame_cnt u8, data_type u8 (0=IMU,1=CartesianHigh,2=CartesianLow), time_type u8
  off 12 : rsvd[12]
  off 24 : crc32 u32
  off 28 : timestamp[8]  (uint64 ns, sensor clock)
  off 36 : data[]
CartesianHigh point (14 B): int32 x,y,z (mm); u8 reflectivity; u8 tag
IMU point (24 B): float gyro_x,y,z (rad/s); float acc_x,y,z (g)
"""
import struct, argparse
from mcap.writer import Writer, CompressionType

LIDAR_PORT = 56301
IMU_PORT = 56401
DT_IMU, DT_CART_HIGH, DT_CART_LOW = 0x00, 0x01, 0x02
HDR = 36  # bytes before data[]

CDR = b"\x00\x01\x00\x00"  # CDR LE representation header (rustlio2 skips these 4)


def encode_custom_msg(sec, nsec, points, timebase=0, lidar_id=0, frame_id=b""):
    """points: list of (offset_time_ns u32, x f32, y f32, z f32, refl u8, tag u8, line u8)."""
    b = bytearray()
    b += struct.pack("<II", sec, nsec)              # header.stamp
    b += struct.pack("<I", len(frame_id) + 1) + frame_id + b"\x00"   # frame_id string
    while len(b) % 4:                               # align to 4 after string
        b += b"\x00"
    while len(b) % 8:                               # align to 8 before u64
        b += b"\x00"
    b += struct.pack("<Q", timebase)                # timebase u64
    b += struct.pack("<I", len(points))             # point_num u32
    b += struct.pack("<B", lidar_id)                # lidar_id u8
    while len(b) % 4:                               # align to 4 (skips rsvd[3])
        b += b"\x00"
    b += struct.pack("<I", len(points))             # points sequence length
    for ot, x, y, z, refl, tag, line in points:
        b += struct.pack("<Ifff BBB x", ot, x, y, z, refl, tag, line)  # 20 B (x=pad)
    return CDR + bytes(b)


def encode_imu(sec, nsec, gyro, accel, frame_id=b""):
    b = bytearray()
    b += struct.pack("<II", sec, nsec)
    b += struct.pack("<I", len(frame_id) + 1) + frame_id + b"\x00"
    while len(b) % 4:
        b += b"\x00"
    while len(b) % 8:
        b += b"\x00"
    b += struct.pack("<4d", 0.0, 0.0, 0.0, 1.0)     # orientation quat (identity)
    b += struct.pack("<9d", *([0.0] * 9))           # orientation_covariance
    b += struct.pack("<3d", *gyro)                  # angular_velocity (rad/s)
    b += struct.pack("<9d", *([0.0] * 9))           # angular_velocity_covariance
    b += struct.pack("<3d", *accel)                 # linear_acceleration (g)
    b += struct.pack("<9d", *([0.0] * 9))           # linear_acceleration_covariance
    return CDR + bytes(b)


def iter_packets(path):
    """Yield (pcap_ts_ns, dst_port, payload) for each UDP packet in a classic pcap."""
    f = open(path, "rb")
    gh = f.read(24)
    if gh[:4] != b"\xd4\xc3\xb2\xa1":
        raise SystemExit(f"unexpected pcap magic {gh[:4].hex()} (expected LE-usec d4c3b2a1)")
    while True:
        h = f.read(16)
        if len(h) < 16:
            break
        ts_sec, ts_usec, incl, _orig = struct.unpack("<IIII", h)
        data = f.read(incl)
        if len(data) < 42:
            continue
        ihl = (data[14] & 0x0F) * 4
        uoff = 14 + ihl
        if uoff + 8 > len(data):
            continue
        dst_port = struct.unpack("!H", data[uoff + 2 : uoff + 4])[0]
        yield ts_sec * 1_000_000_000 + ts_usec * 1000, dst_port, data[uoff + 8 :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap")
    ap.add_argument("out_mcap")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop after this many seconds of sensor data (0 = all)")
    ap.add_argument("--frame-ms", type=float, default=100.0,
                    help="lidar frame window in ms (10 Hz default)")
    ap.add_argument("--compression", choices=["none", "zstd", "lz4"], default="zstd",
                    help="MCAP chunk compression (use 'none' for the C++ harness)")
    args = ap.parse_args()

    comp = {"none": CompressionType.NONE, "zstd": CompressionType.ZSTD,
            "lz4": CompressionType.LZ4}[args.compression]
    w = Writer(open(args.out_mcap, "wb"), compression=comp,
               use_chunking=(comp != CompressionType.NONE))
    w.start(profile="", library="pcap_to_mcap")
    sch_l = w.register_schema(name="livox_ros_driver2/msg/CustomMsg", encoding="ros2msg", data=b"")
    sch_i = w.register_schema(name="sensor_msgs/msg/Imu", encoding="ros2msg", data=b"")
    ch_l = w.register_channel(topic="/livox/lidar", message_encoding="cdr", schema_id=sch_l)
    ch_i = w.register_channel(topic="/livox/imu", message_encoding="cdr", schema_id=sch_i)

    clock_off = None          # epoch_ns - sensor_ns, from first data packet
    frame_ns = int(args.frame_ms * 1e6)
    cur_start = None          # sensor_ts of current frame start
    cur_pts = []
    n_lidar_frames = n_lidar_pts = n_imu = 0
    seq = 0
    t0_sensor = None
    last_sensor = None

    def flush_frame():
        nonlocal n_lidar_frames, seq
        if not cur_pts:
            return
        epoch = cur_start + clock_off
        sec, nsec = divmod(epoch, 1_000_000_000)
        data = encode_custom_msg(sec, nsec, cur_pts)
        w.add_message(channel_id=ch_l, log_time=epoch, data=data, publish_time=epoch, sequence=seq)
        seq += 1
        n_lidar_frames += 1

    for pcap_ts, port, pl in iter_packets(args.pcap):
        if len(pl) < HDR:
            continue
        # Only lidar/imu DATA packets carry the sensor timestamp at off 28; the
        # control/push packets (ports 56000/56201) have a different layout, so
        # derive all timing strictly from the data ports.
        is_lidar = port == LIDAR_PORT and pl[10] in (DT_CART_HIGH, DT_CART_LOW)
        is_imu = port == IMU_PORT and pl[10] == DT_IMU
        if not (is_lidar or is_imu):
            continue
        data_type = pl[10]
        sensor_ts = struct.unpack("<Q", pl[28:36])[0]
        if clock_off is None:
            clock_off = pcap_ts - sensor_ts
            t0_sensor = sensor_ts
        last_sensor = sensor_ts
        if args.max_seconds and (sensor_ts - t0_sensor) > args.max_seconds * 1e9:
            break

        if is_lidar:
            dot_num, time_interval = struct.unpack("<H", pl[5:7])[0], struct.unpack("<H", pl[3:5])[0]
            if cur_start is None:
                cur_start = sensor_ts
            if sensor_ts - cur_start >= frame_ns:
                flush_frame()
                cur_pts.clear()
                cur_start = sensor_ts
            span_ns = time_interval * 100         # 0.1us units -> ns, packet duration
            per_pt = span_ns / dot_num if dot_num else 0.0
            pkt_off = sensor_ts - cur_start
            pts = pl[HDR:]
            if data_type == DT_CART_HIGH:
                psz, sc = 14, 1000.0
                for i in range(dot_num):
                    o = i * psz
                    if o + psz > len(pts):
                        break
                    x, y, z = struct.unpack("<iii", pts[o : o + 12])
                    refl, tag = pts[o + 12], pts[o + 13]
                    ot = int(pkt_off + i * per_pt)
                    cur_pts.append((ot & 0xFFFFFFFF, x / sc, y / sc, z / sc, refl, tag, 0))
                    n_lidar_pts += 1
            else:  # CartesianLow: int16 cm
                psz, sc = 8, 100.0
                for i in range(dot_num):
                    o = i * psz
                    if o + psz > len(pts):
                        break
                    x, y, z = struct.unpack("<hhh", pts[o : o + 6])
                    refl, tag = pts[o + 6], pts[o + 7]
                    ot = int(pkt_off + i * per_pt)
                    cur_pts.append((ot & 0xFFFFFFFF, x / sc, y / sc, z / sc, refl, tag, 0))
                    n_lidar_pts += 1

        else:  # is_imu
            gx, gy, gz, ax, ay, az = struct.unpack("<6f", pl[HDR : HDR + 24])
            epoch = sensor_ts + clock_off
            sec, nsec = divmod(epoch, 1_000_000_000)
            data = encode_imu(sec, nsec, (gx, gy, gz), (ax, ay, az))
            w.add_message(channel_id=ch_i, log_time=epoch, data=data, publish_time=epoch, sequence=seq)
            seq += 1
            n_imu += 1

    flush_frame()
    w.finish()
    dur = (last_sensor - t0_sensor) / 1e9 if t0_sensor and last_sensor else 0.0
    print(f"wrote {args.out_mcap}: {n_lidar_frames} lidar frames ({n_lidar_pts} pts), "
          f"{n_imu} imu samples, {dur:.1f}s")


if __name__ == "__main__":
    main()
