use std::path::Path;
use rustlio::commons::*;
use rustlio::map_builder::{BuilderStatus, MapBuilder};
use rustlio::utils;
use ndarray::Array2;
use ndarray_npy::write_npy;

fn load_config(path: &str) -> Config {
    if Path::new(path).exists() {
        Config::from_yaml_path(path).expect("Failed to parse config")
    } else {
        eprintln!("Config not found at {}, using defaults", path);
        Config::default()
    }
}

fn read_mcap_bag(path: &str, config: &Config) -> Vec<(f64, McapMessage)> {
    let data = std::fs::read(path).expect("Failed to read MCAP file");
    let mut messages = Vec::new();

    for msg_result in mcap::MessageStream::new(&data).expect("Failed to parse MCAP") {
        let msg = match msg_result {
            Ok(m) => m,
            Err(_) => continue,
        };
        let topic = msg.channel.topic.as_str();
        let log_time = msg.log_time as f64 * 1e-9;

        if topic == config.imu_topic {
            if let Some(imu) = parse_imu_cdr(&msg.data) {
                messages.push((log_time, McapMessage::Imu(imu)));
            }
        } else if topic == config.lidar_topic {
            if config.lidar_type != 1 {
                if LIDAR_TYPE_WARN.swap(false, std::sync::atomic::Ordering::Relaxed) {
                    log::warn!(
                        "lidar_type={} is not yet supported by the MCAP reader (only Livox custom msg, type 1); skipping lidar messages",
                        config.lidar_type
                    );
                }
                continue;
            }
            if let Some(cloud_data) = parse_livox_custom_msg(&msg.data, config) {
                messages.push((log_time, McapMessage::Lidar(cloud_data)));
            }
        }
    }
    messages.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    messages
}

enum McapMessage {
    Imu(IMUData),
    Lidar(LidarCloud),
}

struct LidarCloud {
    cloud: PointCloud,
    start_time: f64,
    end_time: f64,
}

static IMU_DEBUG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static LIDAR_TYPE_WARN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

fn parse_imu_cdr(data: &[u8]) -> Option<IMUData> {
    if data.len() < 4 + 8 + 8 + 8 * 10 {
        return None;
    }
    let buf = &data[4..]; // skip CDR header
    let sec = u32::from_le_bytes(buf[0..4].try_into().ok()?);
    let nsec = u32::from_le_bytes(buf[4..8].try_into().ok()?);
    let time = utils::stamp_to_sec(sec, nsec);

    // Skip header frame_id string
    let fid_len = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;
    let mut offset = 12 + fid_len;
    offset = (offset + 3) & !3; // align to 4

    if log::log_enabled!(log::Level::Trace)
        && IMU_DEBUG.swap(false, std::sync::atomic::Ordering::Relaxed)
    {
        let fid = std::str::from_utf8(&buf[12..12 + fid_len.saturating_sub(1)]).unwrap_or("?");
        log::trace!("IMU CDR debug: data_len={}, buf_len={}, fid_len={}, fid='{}', offset_after_fid={}",
            data.len(), buf.len(), fid_len, fid, offset);
        // Print all f64 values from offset onward
        let mut o = (offset + 7) & !7;
        log::trace!("  aligned offset={}", o);
        let mut idx = 0;
        while o + 8 <= buf.len() {
            let v = f64::from_le_bytes(buf[o..o+8].try_into().ok()?);
            if idx < 30 {
                log::trace!("  f64[{}] @ {}: {:.8}", idx, o, v);
            }
            o += 8;
            idx += 1;
        }
    }

    // Orientation: quaternion (4 x f64) - skip
    offset = (offset + 7) & !7; // align to 8
    offset += 4 * 8; // skip quaternion
    offset += 9 * 8; // skip orientation_covariance

    // Angular velocity: x, y, z
    let gx = f64::from_le_bytes(buf[offset..offset + 8].try_into().ok()?);
    let gy = f64::from_le_bytes(buf[offset + 8..offset + 16].try_into().ok()?);
    let gz = f64::from_le_bytes(buf[offset + 16..offset + 24].try_into().ok()?);
    offset += 3 * 8;
    offset += 9 * 8; // skip angular_velocity_covariance

    // Linear acceleration: x, y, z
    let ax = f64::from_le_bytes(buf[offset..offset + 8].try_into().ok()?);
    let ay = f64::from_le_bytes(buf[offset + 8..offset + 16].try_into().ok()?);
    let az = f64::from_le_bytes(buf[offset + 16..offset + 24].try_into().ok()?);

    Some(IMUData {
        // Livox IMU reports linear acceleration in units of g; convert to m/s^2
        // with the standard gravity constant so |acc| at rest matches the
        // gravity state initialised at IESKF::GRAVITY (9.81). The old `* 10.0`
        // left a ~0.19 m/s^2 vertical residual (10 - 9.81) that integrated into
        // z-drift. See faithful_check.
        acc: V3D::new(ax, ay, az) * 9.80665,
        gyro: V3D::new(gx, gy, gz),
        time,
    })
}

fn parse_livox_custom_msg(data: &[u8], config: &Config) -> Option<LidarCloud> {
    if data.len() < 30 {
        return None;
    }
    let buf = &data[4..]; // skip CDR header

    // Header: stamp
    let sec = u32::from_le_bytes(buf[0..4].try_into().ok()?);
    let nsec = u32::from_le_bytes(buf[4..8].try_into().ok()?);
    let header_time = utils::stamp_to_sec(sec, nsec) + config.time_offset_lidar_to_imu;

    // Skip frame_id string
    let fid_len = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;
    let mut offset = 12 + fid_len;
    offset = (offset + 3) & !3;

    // timebase: u64, point_num: u32, lidar_id: u8
    offset = (offset + 7) & !7;
    if offset + 13 > buf.len() {
        return None;
    }
    let _timebase = u64::from_le_bytes(buf[offset..offset + 8].try_into().ok()?);
    offset += 8;
    let point_num = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?) as usize;
    offset += 4;
    let _lidar_id = buf[offset];
    offset += 1;
    offset = (offset + 3) & !3;

    // Points array: sequence length
    if offset + 4 > buf.len() {
        return None;
    }
    let seq_len = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?) as usize;
    offset += 4;

    // Each CustomPoint: offset_time(u32) + x(f32) + y(f32) + z(f32) + reflectivity(u8) + tag(u8) + line(u8)
    let point_size = 20; // 19 bytes + 1 padding (CDR aligns struct elements to max member alignment = 4)
    let filter_num = config.lidar_filter_num as usize;

    let mut cloud = Vec::new();
    let mut max_offset_time: f32 = 0.0;
    let mut prev_xyz: Option<(f32, f32, f32)> = None;

    let mut i = 0;
    while i < seq_len.min(point_num) {
        let po = offset + i * point_size;
        if po + point_size > buf.len() {
            break;
        }
        let offset_time = u32::from_le_bytes(buf[po..po + 4].try_into().ok()?);
        let x = f32::from_le_bytes(buf[po + 4..po + 8].try_into().ok()?);
        let y = f32::from_le_bytes(buf[po + 8..po + 12].try_into().ok()?);
        let z = f32::from_le_bytes(buf[po + 12..po + 16].try_into().ok()?);
        let reflectivity = buf[po + 16];
        let tag = buf[po + 17];
        let line = buf[po + 18];

        if utils::livox_point_valid(tag, line)
            && !utils::livox_is_duplicate(prev_xyz, x, y, z)
        {
            if let Some(p) = utils::livox_to_point(
                x, y, z, reflectivity, offset_time,
                config.lidar_min_range, config.lidar_max_range,
            ) {
                if p.curvature > max_offset_time {
                    max_offset_time = p.curvature;
                }
                cloud.push(p);
                prev_xyz = Some((x, y, z));
            }
        }
        i += filter_num;
    }

    Some(LidarCloud {
        cloud,
        start_time: header_time,
        end_time: header_time + max_offset_time as f64 / 1000.0,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("../config_examples/mid360.yaml");
    let bag_path = args.get(2).map(|s| s.as_str()).unwrap_or(
        "../data/ruwik2_pt3_bag_custom/ruwik2_pt3_bag_custom_0.mcap",
    );
    let output_path = args.get(3).map(|s| s.as_str());
    let duration_limit: f64 = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60.0);

    let config = load_config(config_path);
    rustlio::logging::init(config.log_level);
    log::info!("Loading config from: {}", config_path);

    log::info!("Reading MCAP bag: {}", bag_path);
    let messages = read_mcap_bag(bag_path, &config);
    let imu_count = messages.iter().filter(|(_, m)| matches!(m, McapMessage::Imu(_))).count();
    let lidar_count = messages.iter().filter(|(_, m)| matches!(m, McapMessage::Lidar(_))).count();
    log::info!("Loaded {} messages ({} IMU, {} LiDAR)", messages.len(), imu_count, lidar_count);

    log::debug!("Duration limit: {:.0}s", duration_limit);

    let mut builder = MapBuilder::new(config);

    // Split the time-sorted stream into IMU samples and lidar frames, then pair
    // them with the same syncPackage rule as the C++ reference: a lidar frame
    // gets every IMU sample up to its cloud_end_time. This is essential — a
    // frame's stamp is its START time, so the IMU spanning the frame's own sweep
    // arrive AFTER it in the stream. The previous code drained the IMU buffer on
    // each lidar message, giving every frame the prior frame's IMU window and
    // forcing the undistort to extrapolate ~0.1 s of motion per frame, which
    // accumulated into large vertical (z) drift.
    let imus: Vec<&IMUData> = messages
        .iter()
        .filter_map(|(_, m)| if let McapMessage::Imu(i) = m { Some(i) } else { None })
        .collect();
    let frames: Vec<&LidarCloud> = messages
        .iter()
        .filter_map(|(_, m)| if let McapMessage::Lidar(l) = m { Some(l) } else { None })
        .collect();

    let mut odom_count = 0;
    let mut first_odom_time: Option<f64> = None;
    let mut odom_records: Vec<[f64; 7]> = Vec::new();
    let mut imu_idx = 0usize;

    for lidar in frames {
        if let Some(t0) = first_odom_time {
            if lidar.start_time - t0 > duration_limit {
                break;
            }
        }

        let mut cloud = lidar.cloud.clone();
        cloud.sort_by(|a, b| a.curvature.partial_cmp(&b.curvature).unwrap());
        let cloud_end = lidar.end_time;

        let mut pkg_imus = Vec::new();
        while imu_idx < imus.len() && imus[imu_idx].time < cloud_end {
            pkg_imus.push(imus[imu_idx].clone());
            imu_idx += 1;
        }
        if pkg_imus.is_empty() {
            continue;
        }

        let mut package = SyncPackage {
            imus: pkg_imus,
            cloud,
            cloud_start_time: lidar.start_time,
            cloud_end_time: cloud_end,
        };
        builder.process(&mut package);

        if builder.status() == BuilderStatus::Mapping {
            let state = &builder.kf.x;
            let speed = state.v.norm();

            if first_odom_time.is_none() {
                first_odom_time = Some(lidar.start_time);
            }

            odom_records.push([
                lidar.start_time,
                state.imu_to_world_trans[0], state.imu_to_world_trans[1], state.imu_to_world_trans[2],
                state.v[0], state.v[1], state.v[2],
            ]);

            if odom_count % 100 == 0 {
                log::debug!(
                    "odom {}: speed={:.3} m/s  t={:.2}s",
                    odom_count, speed,
                    lidar.start_time - first_odom_time.unwrap_or(lidar.start_time),
                );
            }
            odom_count += 1;
        }
    }

    log::info!("Processing complete. {} odom outputs.", odom_count);

    if let Some(out) = output_path {
        let rows = odom_records.len();
        let mut data = Array2::<f64>::zeros((rows, 7));
        for (i, rec) in odom_records.iter().enumerate() {
            for j in 0..7 {
                data[[i, j]] = rec[j];
            }
        }
        write_npy(out, &data).expect("Failed to write .npy");
        log::info!("Saved {} odom records to {}", rows, out);
    }
}
