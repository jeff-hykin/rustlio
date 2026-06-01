use std::path::Path;
use fastlio_rs::commons::*;
use fastlio_rs::map_builder::{BuilderStatus, MapBuilder};
use fastlio_rs::utils;

fn load_config(path: &str) -> Config {
    if Path::new(path).exists() {
        Config::from_yaml_path(path).expect("Failed to parse config")
    } else {
        eprintln!("Config not found at {}, using defaults", path);
        Config::default()
    }
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

fn parse_imu_cdr(data: &[u8]) -> Option<IMUData> {
    if data.len() < 4 + 8 + 8 + 8 * 10 {
        return None;
    }
    let buf = &data[4..];
    let sec = u32::from_le_bytes(buf[0..4].try_into().ok()?);
    let nsec = u32::from_le_bytes(buf[4..8].try_into().ok()?);
    let time = utils::stamp_to_sec(sec, nsec);

    let fid_len = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;
    let mut offset = 12 + fid_len;
    offset = (offset + 3) & !3;

    offset = (offset + 7) & !7;
    offset += 4 * 8;
    offset += 9 * 8;

    let gx = f64::from_le_bytes(buf[offset..offset + 8].try_into().ok()?);
    let gy = f64::from_le_bytes(buf[offset + 8..offset + 16].try_into().ok()?);
    let gz = f64::from_le_bytes(buf[offset + 16..offset + 24].try_into().ok()?);
    offset += 3 * 8;
    offset += 9 * 8;

    let ax = f64::from_le_bytes(buf[offset..offset + 8].try_into().ok()?);
    let ay = f64::from_le_bytes(buf[offset + 8..offset + 16].try_into().ok()?);
    let az = f64::from_le_bytes(buf[offset + 16..offset + 24].try_into().ok()?);

    Some(IMUData {
        acc: V3D::new(ax, ay, az) * 10.0,
        gyro: V3D::new(gx, gy, gz),
        time,
    })
}

fn parse_livox_custom_msg(data: &[u8], config: &Config) -> Option<LidarCloud> {
    if data.len() < 30 {
        return None;
    }
    let buf = &data[4..];

    let sec = u32::from_le_bytes(buf[0..4].try_into().ok()?);
    let nsec = u32::from_le_bytes(buf[4..8].try_into().ok()?);
    let header_time = utils::stamp_to_sec(sec, nsec);

    let fid_len = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;
    let mut offset = 12 + fid_len;
    offset = (offset + 3) & !3;

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

    if offset + 4 > buf.len() {
        return None;
    }
    let seq_len = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?) as usize;
    offset += 4;

    let point_size = 20;
    let filter_num = config.lidar_filter_num as usize;

    let mut cloud = Vec::new();
    let mut max_offset_time: f32 = 0.0;

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

        if utils::livox_point_valid(tag, line) {
            if let Some(p) = utils::livox_to_point(
                x, y, z, reflectivity, offset_time,
                config.lidar_min_range, config.lidar_max_range,
            ) {
                if p.curvature > max_offset_time {
                    max_offset_time = p.curvature;
                }
                cloud.push(p);
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
            if config.lidar_type == 1 {
                if let Some(cloud_data) = parse_livox_custom_msg(&msg.data, config) {
                    messages.push((log_time, McapMessage::Lidar(cloud_data)));
                }
            }
        }
    }
    messages.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    messages
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let config_path = args.get(1).map(|s| s.as_str())
        .unwrap_or("../fastlio2/config/lio.yaml");
    let bag_path = args.get(2).map(|s| s.as_str())
        .unwrap_or("../data/ruwik2_pt3_bag_custom/ruwik2_pt3_bag_custom_0.mcap");
    let output_path = args.get(3).map(|s| s.as_str());
    let save_mode = output_path.is_some();

    println!("Loading config from: {}", config_path);
    let config = load_config(config_path);

    println!("Reading MCAP bag: {}", bag_path);
    let messages = read_mcap_bag(bag_path, &config);
    println!("Loaded {} messages", messages.len());

    let rec = if let Some(path) = output_path {
        println!("Saving to: {}", path);
        rerun::RecordingStreamBuilder::new("fastlio2").save(path)?
    } else {
        println!("Opening Rerun viewer...");
        rerun::RecordingStreamBuilder::new("fastlio2").spawn()?
    };

    let mut builder = MapBuilder::new(config);
    let mut imu_buf: Vec<IMUData> = Vec::new();
    let mut odom_count = 0;
    let mut trajectory: Vec<[f32; 3]> = Vec::new();
    let mut first_time: Option<f64> = None;

    for (_log_time, msg) in &messages {
        match msg {
            McapMessage::Imu(imu) => {
                imu_buf.push(imu.clone());
            }
            McapMessage::Lidar(lidar) => {
                if imu_buf.is_empty() {
                    continue;
                }

                let t = lidar.start_time;
                if first_time.is_none() {
                    first_time = Some(t);
                }
                let t_rel = t - first_time.unwrap();

                rec.set_time_sequence("frame", odom_count as i64);
                rec.set_duration_secs("time", t_rel);

                // Log raw LiDAR point cloud
                let lidar_pts: Vec<[f32; 3]> = lidar.cloud.iter()
                    .map(|p| [p.x, p.y, p.z])
                    .collect();
                let intensities: Vec<f32> = lidar.cloud.iter()
                    .map(|p| p.intensity)
                    .collect();
                rec.log(
                    "world/lidar/raw",
                    &rerun::Points3D::new(&lidar_pts)
                        .with_radii([0.01f32])
                        .with_colors(intensities.iter().map(|&i| {
                            let v = (i.clamp(0.0, 255.0)) as u8;
                            rerun::Color::from_rgb(v, v, 255 - v)
                        })),
                )?;

                let mut package = SyncPackage {
                    imus: imu_buf.drain(..).collect(),
                    cloud: lidar.cloud.clone(),
                    cloud_start_time: lidar.start_time,
                    cloud_end_time: lidar.end_time,
                };
                builder.process(&mut package);

                if builder.status() == BuilderStatus::Mapping {
                    let state = &builder.kf.x;
                    let pos = [state.t_wi[0] as f32, state.t_wi[1] as f32, state.t_wi[2] as f32];
                    let speed = state.v.norm();

                    // Robot position
                    let rot = &state.r_wi;
                    let q = rotation_matrix_to_quaternion(rot);
                    rec.log(
                        "world/robot",
                        &rerun::Transform3D::from_translation_rotation(
                            pos,
                            rerun::Quaternion::from_xyzw(q),
                        ),
                    )?;

                    rec.log(
                        "world/robot/origin",
                        &rerun::Points3D::new([[0.0f32, 0.0, 0.0]])
                            .with_radii([0.1])
                            .with_colors([rerun::Color::from_rgb(255, 50, 50)]),
                    )?;

                    // Trajectory
                    trajectory.push(pos);
                    if trajectory.len() > 1 {
                        rec.log(
                            "world/trajectory",
                            &rerun::LineStrips3D::new([&trajectory[..]])
                                .with_colors([rerun::Color::from_rgb(0, 255, 100)])
                                .with_radii([0.03]),
                        )?;
                    }

                    // Speed scalar
                    rec.log("metrics/speed", &rerun::Scalars::single(speed))?;

                    // Velocity components
                    rec.log("metrics/vx", &rerun::Scalars::single(state.v[0]))?;
                    rec.log("metrics/vy", &rerun::Scalars::single(state.v[1]))?;
                    rec.log("metrics/vz", &rerun::Scalars::single(state.v[2]))?;

                    if odom_count % 100 == 0 {
                        println!("frame {}: pos=({:.2}, {:.2}, {:.2}) speed={:.2} m/s  t={:.1}s",
                            odom_count, pos[0], pos[1], pos[2], speed, t_rel);
                    }
                    odom_count += 1;
                }
            }
        }
    }

    if save_mode {
        println!("\nDone. {} frames saved to {}", odom_count, output_path.unwrap());
    } else {
        println!("\nDone. {} frames sent to Rerun viewer.", odom_count);
    }
    Ok(())
}

fn rotation_matrix_to_quaternion(r: &nalgebra::Matrix3<f64>) -> [f32; 4] {
    let trace = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
    let (w, x, y, z) = if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        (0.25 / s, (r[(2, 1)] - r[(1, 2)]) * s, (r[(0, 2)] - r[(2, 0)]) * s, (r[(1, 0)] - r[(0, 1)]) * s)
    } else if r[(0, 0)] > r[(1, 1)] && r[(0, 0)] > r[(2, 2)] {
        let s = 2.0 * (1.0 + r[(0, 0)] - r[(1, 1)] - r[(2, 2)]).sqrt();
        ((r[(2, 1)] - r[(1, 2)]) / s, 0.25 * s, (r[(0, 1)] + r[(1, 0)]) / s, (r[(0, 2)] + r[(2, 0)]) / s)
    } else if r[(1, 1)] > r[(2, 2)] {
        let s = 2.0 * (1.0 + r[(1, 1)] - r[(0, 0)] - r[(2, 2)]).sqrt();
        ((r[(0, 2)] - r[(2, 0)]) / s, (r[(0, 1)] + r[(1, 0)]) / s, 0.25 * s, (r[(1, 2)] + r[(2, 1)]) / s)
    } else {
        let s = 2.0 * (1.0 + r[(2, 2)] - r[(0, 0)] - r[(1, 1)]).sqrt();
        ((r[(1, 0)] - r[(0, 1)]) / s, (r[(0, 2)] + r[(2, 0)]) / s, (r[(1, 2)] + r[(2, 1)]) / s, 0.25 * s)
    };
    [x as f32, y as f32, z as f32, w as f32]
}
