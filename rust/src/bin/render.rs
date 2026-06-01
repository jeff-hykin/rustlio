//! `render` — run the LIO pipeline directly on a raw Livox mid360 `.pcap`
//! (SDK2 UDP capture) and write a Rerun `.rrd` containing the estimated
//! trajectory and the odom-adjusted world point cloud.
//!
//! Usage:
//!   render <input.pcap> <output.rrd> [config.yaml] [duration_s]
//!
//! The pcap is a classic libpcap capture of Livox SDK2 traffic:
//!   * point cloud  -> UDP dst port 56301, data_type 1 (Cartesian high, mm)
//!   * IMU          -> UDP dst port 56401, data_type 0 (gyro rad/s, acc in g)
//! Both carry a 36-byte LivoxLidarEthernetPacket header with a device
//! nanosecond timestamp. Points/IMU are streamed (the file can be many GB),
//! assembled into ~10 Hz frames, and fed through `MapBuilder`.

use std::io::{BufReader, Read};
use std::fs::File;

use fastlio_rs::commons::*;
use fastlio_rs::map_builder::{BuilderStatus, MapBuilder};

const CLOUD_PORT: u16 = 56301;
const IMU_PORT: u16 = 56401;
const LIVOX_HEADER: usize = 36; // LivoxLidarEthernetPacket header (packed)
const CLOUD_POINT_SIZE: usize = 14; // int32 x,y,z (mm) + u8 reflectivity + u8 tag
const GRAVITY: f64 = 9.81; // Livox IMU reports accel in g
const FRAME_SEC: f64 = 0.1; // assemble LiDAR frames at ~10 Hz

struct RawLidarPoint {
    x: f32,
    y: f32,
    z: f32,
    intensity: f32,
    abs_time: f64, // seconds (device clock)
}

/// Minimal streaming reader for a classic (LE, microsecond) libpcap file that
/// yields Livox SDK2 UDP payloads via callbacks.
struct PcapReader<R: Read> {
    r: R,
}

impl<R: Read> PcapReader<R> {
    fn new(mut r: R) -> std::io::Result<Self> {
        let mut gh = [0u8; 24];
        r.read_exact(&mut gh)?;
        let magic = u32::from_le_bytes(gh[0..4].try_into().unwrap());
        if magic != 0xa1b2_c3d4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("not a little-endian microsecond pcap (magic {magic:#x})"),
            ));
        }
        Ok(PcapReader { r })
    }

    /// Read the next record into `buf`; returns the UDP (dst_port, payload range)
    /// for IPv4/UDP packets, or `Some(None)` for non-UDP records, `None` at EOF.
    fn next_packet<'a>(&mut self, buf: &'a mut Vec<u8>) -> Option<(u16, &'a [u8])> {
        let mut rh = [0u8; 16];
        if self.r.read_exact(&mut rh).is_err() {
            return None;
        }
        let incl = u32::from_le_bytes(rh[8..12].try_into().unwrap()) as usize;
        buf.resize(incl, 0);
        if self.r.read_exact(buf).is_err() {
            return None;
        }
        // Ethernet (14) + IPv4 + UDP (8). Return a sentinel port 0 to skip.
        if incl < 42 {
            return Some((0, &[]));
        }
        let eth_type = u16::from_be_bytes([buf[12], buf[13]]);
        if eth_type != 0x0800 {
            return Some((0, &[]));
        }
        let ihl = ((buf[14] & 0x0f) as usize) * 4;
        let proto = buf[14 + 9];
        let udp = 14 + ihl;
        if proto != 17 || buf.len() < udp + 8 {
            return Some((0, &[]));
        }
        let dport = u16::from_be_bytes([buf[udp + 2], buf[udp + 3]]);
        let ulen = u16::from_be_bytes([buf[udp + 4], buf[udp + 5]]) as usize;
        let start = udp + 8;
        let end = (udp + ulen).min(buf.len());
        if end <= start {
            return Some((0, &[]));
        }
        Some((dport, &buf[start..end]))
    }
}

fn livox_timestamp_ns(payload: &[u8]) -> u64 {
    u64::from_le_bytes(payload[28..36].try_into().unwrap())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: render <input.pcap> <output.rrd> [config.yaml] [duration_s]");
        std::process::exit(2);
    }
    let pcap_path = &args[1];
    let out_path = &args[2];
    let config_path = args.get(3).map(|s| s.as_str()).unwrap_or("../config_examples/mid360.yaml");
    let duration_limit: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0); // 0 = whole file

    let config = if std::path::Path::new(config_path).exists() {
        Config::from_yaml_path(config_path)?
    } else {
        eprintln!("config {config_path} not found, using defaults");
        Config::default()
    };
    println!("Config: {config_path} (filter_num={}, min/max range={}/{})",
        config.lidar_filter_num, config.lidar_min_range, config.lidar_max_range);

    let rec = rerun::RecordingStreamBuilder::new("fastlio2_render").save(out_path)?;

    let mut builder = MapBuilder::new(config.clone());
    let mut imu_buf: Vec<IMUData> = Vec::new();
    let mut frame_pts: Vec<RawLidarPoint> = Vec::new();
    let mut frame_start: Option<f64> = None;
    let mut filter_ctr: usize = 0;
    let filter_num = config.lidar_filter_num.max(1) as usize;

    let mut trajectory: Vec<[f32; 3]> = Vec::new();
    let mut world_map: Vec<Point> = Vec::new();
    let mut first_frame_time: Option<f64> = None;
    let mut frame_idx: i64 = 0;
    let mut imu_count: u64 = 0;
    let mut done = false;

    let file = File::open(pcap_path)?;
    let mut reader = PcapReader::new(BufReader::with_capacity(1 << 20, file))?;
    let mut buf: Vec<u8> = Vec::with_capacity(2048);

    println!("Reading {pcap_path} ...");
    while let Some((dport, payload)) = reader.next_packet(&mut buf) {
        if done {
            break;
        }
        if payload.len() < LIVOX_HEADER {
            continue;
        }
        let ts_ns = livox_timestamp_ns(payload);
        let ts = ts_ns as f64 * 1e-9;
        let body = &payload[LIVOX_HEADER..];

        if dport == IMU_PORT {
            if body.len() < 24 {
                continue;
            }
            let g = |o: usize| f32::from_le_bytes(body[o..o + 4].try_into().unwrap()) as f64;
            imu_buf.push(IMUData {
                gyro: V3D::new(g(0), g(4), g(8)),
                acc: V3D::new(g(12), g(16), g(20)) * GRAVITY,
                time: ts,
            });
            imu_count += 1;
            continue;
        }

        if dport != CLOUD_PORT {
            continue;
        }

        // Cartesian-high points: int32 x,y,z (mm) + u8 reflectivity + u8 tag.
        let dot_num = u16::from_le_bytes([payload[5], payload[6]]) as usize;
        let interval_ns = u16::from_le_bytes([payload[3], payload[4]]) as f64 * 100.0; // total span (ns)
        let pt_dt = if dot_num > 0 { interval_ns / dot_num as f64 * 1e-9 } else { 0.0 };

        for i in 0..dot_num {
            let o = i * CLOUD_POINT_SIZE;
            if o + CLOUD_POINT_SIZE > body.len() {
                break;
            }
            let xi = i32::from_le_bytes(body[o..o + 4].try_into().unwrap());
            let yi = i32::from_le_bytes(body[o + 4..o + 8].try_into().unwrap());
            let zi = i32::from_le_bytes(body[o + 8..o + 12].try_into().unwrap());
            if xi == 0 && yi == 0 && zi == 0 {
                continue; // dropout
            }
            let refl = body[o + 12];
            let x = xi as f64 / 1000.0;
            let y = yi as f64 / 1000.0;
            let z = zi as f64 / 1000.0;
            let r2 = x * x + y * y + z * z;
            if r2 < config.lidar_min_range * config.lidar_min_range
                || r2 > config.lidar_max_range * config.lidar_max_range
            {
                continue;
            }
            filter_ctr += 1;
            if filter_ctr % filter_num != 0 {
                continue;
            }
            let abs_time = ts + i as f64 * pt_dt;
            if frame_start.is_none() {
                frame_start = Some(abs_time);
            }
            frame_pts.push(RawLidarPoint { x: x as f32, y: y as f32, z: z as f32, intensity: refl as f32, abs_time });
        }

        // Close a frame once it spans FRAME_SEC.
        let fs = match frame_start {
            Some(v) => v,
            None => continue,
        };
        if ts - fs < FRAME_SEC {
            continue;
        }

        let frame_end = frame_pts.last().map(|p| p.abs_time).unwrap_or(ts);
        // Per-frame cloud in LiDAR frame, curvature = ms since frame start.
        let cloud: PointCloud = frame_pts
            .iter()
            .map(|p| Point {
                x: p.x,
                y: p.y,
                z: p.z,
                intensity: p.intensity,
                curvature: ((p.abs_time - fs) * 1000.0) as f32,
            })
            .collect();
        frame_pts.clear();
        frame_start = None;

        if imu_buf.is_empty() || cloud.is_empty() {
            continue;
        }

        // IMU up to frame_end goes with this scan; keep the rest for next frame.
        let (this_imus, rest): (Vec<_>, Vec<_>) =
            imu_buf.drain(..).partition(|s| s.time <= frame_end);
        imu_buf = rest;
        if this_imus.is_empty() {
            continue; // no IMU spanning this scan yet; keep accumulating
        }

        let mut package = SyncPackage {
            imus: this_imus,
            cloud,
            cloud_start_time: fs,
            cloud_end_time: frame_end,
        };
        builder.process(&mut package);

        if builder.status() != BuilderStatus::Mapping {
            continue;
        }

        let st = &builder.kf.x;
        let r_wl = st.imu_to_world_rot * st.lidar_to_imu_rot;
        let t_wl = st.imu_to_world_trans + st.imu_to_world_rot * st.lidar_to_imu_trans;
        let pos = [st.imu_to_world_trans[0] as f32, st.imu_to_world_trans[1] as f32, st.imu_to_world_trans[2] as f32];

        if first_frame_time.is_none() {
            first_frame_time = Some(fs);
        }
        let t_rel = fs - first_frame_time.unwrap();
        if duration_limit > 0.0 && t_rel > duration_limit {
            done = true;
        }

        // Odom-adjusted (world) points for this scan.
        let world_scan: Vec<Point> = package
            .cloud
            .iter()
            .map(|p| {
                let pv = V3D::new(p.x as f64, p.y as f64, p.z as f64);
                let pw = r_wl * pv + t_wl;
                Point { x: pw[0] as f32, y: pw[1] as f32, z: pw[2] as f32, intensity: p.intensity, curvature: 0.0 }
            })
            .collect();

        rec.set_time_sequence("frame", frame_idx);
        rec.set_duration_secs("time", t_rel);

        // Current scan (this frame) in world frame.
        let scan_xyz: Vec<[f32; 3]> = world_scan.iter().map(|p| [p.x, p.y, p.z]).collect();
        rec.log(
            "world/scan",
            &rerun::Points3D::new(&scan_xyz)
                .with_radii([0.02f32])
                .with_colors(world_scan.iter().map(|p| intensity_color(p.intensity))),
        )?;

        // Robot pose + trajectory.
        trajectory.push(pos);
        rec.log("world/robot", &rerun::Points3D::new([pos]).with_radii([0.12]).with_colors([rerun::Color::from_rgb(255, 50, 50)]))?;
        if trajectory.len() > 1 {
            rec.log(
                "world/trajectory",
                &rerun::LineStrips3D::new([trajectory.as_slice()])
                    .with_colors([rerun::Color::from_rgb(0, 255, 100)])
                    .with_radii([0.03]),
            )?;
        }

        // Accumulate the world map (bounded via periodic voxel downsample).
        world_map.extend(world_scan);
        if frame_idx % 20 == 19 {
            world_map = fastlio_rs::voxel_grid::downsample(&world_map, config.map_resolution);
        }

        if frame_idx % 50 == 0 {
            println!("frame {frame_idx}: t={t_rel:.1}s pos=({:.2},{:.2},{:.2}) scan={} map={}",
                pos[0], pos[1], pos[2], scan_xyz.len(), world_map.len());
        }
        frame_idx += 1;
    }

    // Final accumulated odom-adjusted world cloud.
    world_map = fastlio_rs::voxel_grid::downsample(&world_map, config.map_resolution);
    let map_xyz: Vec<[f32; 3]> = world_map.iter().map(|p| [p.x, p.y, p.z]).collect();
    rec.log_static(
        "world/cloud",
        &rerun::Points3D::new(&map_xyz)
            .with_radii([0.02f32])
            .with_colors(world_map.iter().map(|p| intensity_color(p.intensity))),
    )?;
    if trajectory.len() > 1 {
        rec.log_static(
            "world/trajectory",
            &rerun::LineStrips3D::new([trajectory.as_slice()])
                .with_colors([rerun::Color::from_rgb(0, 255, 100)])
                .with_radii([0.03]),
        )?;
    }

    println!("\nDone. {frame_idx} frames, {imu_count} IMU samples.");
    println!("World cloud: {} points -> {out_path}", map_xyz.len());
    Ok(())
}

/// Map Livox reflectivity (0..255) to a blue->yellow color.
fn intensity_color(i: f32) -> rerun::Color {
    let v = i.clamp(0.0, 255.0) as u8;
    rerun::Color::from_rgb(v, v, 255 - v)
}
