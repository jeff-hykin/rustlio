//! End-to-end integration tests driving the full `MapBuilder` pipeline
//! (IMU init -> map init -> mapping) with synthetic IMU + LiDAR data.
//!
//! The synthetic world is a static planar "room"; a stationary sensor should
//! stay put, sensor dropouts must not crash the filter, and the velocity-cap
//! guardrail must keep odometry bounded even when the IMU glitches.

use fastlio2::commons::*;
use fastlio2::map_builder::{BuilderStatus, MapBuilder};

const IMU_PERIOD_SECONDS: f64 = 0.01; // 100 Hz IMU
const IMUS_PER_FRAME: usize = 10; // -> 10 Hz LiDAR frames
const FRAME_PERIOD_SECONDS: f64 = IMU_PERIOD_SECONDS * IMUS_PER_FRAME as f64;

fn plane_point(x: f32, y: f32, z: f32) -> Point {
    Point { x, y, z, intensity: 80.0, curvature: 0.0 }
}

/// Dense planar room (floor, ceiling, four walls), all inside LiDAR range so
/// every surface gives the scan matcher well-conditioned plane constraints.
fn room_cloud() -> PointCloud {
    let mut points = Vec::new();
    let spacing = 0.3_f32;
    let half_width = 6.0_f32;
    let half_height = 2.0_f32;

    // floor (z = -2) and ceiling (z = +2)
    let mut x = -half_width;
    while x <= half_width {
        let mut y = -half_width;
        while y <= half_width {
            points.push(plane_point(x, y, -half_height));
            points.push(plane_point(x, y, half_height));
            y += spacing;
        }
        x += spacing;
    }
    // four walls (x = +-6, y = +-6)
    let mut along_wall = -half_width;
    while along_wall <= half_width {
        let mut z = -half_height;
        while z <= half_height {
            points.push(plane_point(-half_width, along_wall, z));
            points.push(plane_point(half_width, along_wall, z));
            points.push(plane_point(along_wall, -half_width, z));
            points.push(plane_point(along_wall, half_width, z));
            z += spacing;
        }
        along_wall += spacing;
    }
    points
}

struct Frame {
    imus: Vec<IMUData>,
    cloud: PointCloud,
    start_time: f64,
    end_time: f64,
}

/// Build `frame_count` frames of the static room. `extra_acceleration(frame_index)`
/// adds body-frame acceleration on top of gravity (zero => stationary sensor).
fn make_frames<F: Fn(usize) -> V3D>(frame_count: usize, extra_acceleration: F) -> Vec<Frame> {
    let cloud = room_cloud();
    let mut frames = Vec::with_capacity(frame_count);
    let mut frame_start = 0.0;
    for frame_index in 0..frame_count {
        let extra = extra_acceleration(frame_index);
        let mut imus = Vec::with_capacity(IMUS_PER_FRAME);
        for sample_index in 0..IMUS_PER_FRAME {
            imus.push(IMUData {
                acc: V3D::new(extra.x, extra.y, 9.81 + extra.z),
                gyro: V3D::zeros(),
                time: frame_start + sample_index as f64 * IMU_PERIOD_SECONDS,
            });
        }
        frames.push(Frame {
            imus,
            cloud: cloud.clone(),
            start_time: frame_start,
            end_time: frame_start + FRAME_PERIOD_SECONDS,
        });
        frame_start += FRAME_PERIOD_SECONDS;
    }
    frames
}

/// Drive every frame through the builder, collecting the reported odom speed for
/// each frame that produced output (i.e. once the builder is in `Mapping`).
fn run(builder: &mut MapBuilder, frames: &[Frame]) -> Vec<f64> {
    let mut speeds = Vec::new();
    for frame in frames {
        let mut package = SyncPackage {
            imus: frame.imus.clone(),
            cloud: frame.cloud.clone(),
            cloud_start_time: frame.start_time,
            cloud_end_time: frame.end_time,
        };
        builder.process(&mut package);
        if builder.status() == BuilderStatus::Mapping {
            speeds.push(builder.kf.x.v.norm());
        }
    }
    speeds
}

#[test]
fn stationary_sensor_reaches_mapping_and_stays_put() {
    let frames = make_frames(20, |_| V3D::zeros());
    let mut builder = MapBuilder::new(Config::default());
    let speeds = run(&mut builder, &frames);

    assert_eq!(builder.status(), BuilderStatus::Mapping, "pipeline never reached Mapping");
    assert!(!speeds.is_empty(), "no odometry emitted while Mapping");

    let max_speed = speeds.iter().cloned().fold(0.0_f64, f64::max);
    assert!(max_speed < 0.5, "stationary sensor drifted: max speed {max_speed} m/s");

    let distance_from_origin = builder.kf.x.t_wi.norm();
    assert!(distance_from_origin < 0.5, "stationary sensor wandered {distance_from_origin} m");
}

#[test]
fn empty_cloud_frame_does_not_break_pipeline() {
    let mut frames = make_frames(20, |_| V3D::zeros());
    frames[12].cloud.clear(); // mid-stream LiDAR dropout

    let mut builder = MapBuilder::new(Config::default());
    let speeds = run(&mut builder, &frames); // must not panic

    assert_eq!(builder.status(), BuilderStatus::Mapping);
    assert!(
        speeds.iter().all(|speed| speed.is_finite()),
        "non-finite odom speed after empty-cloud frame"
    );
}

#[test]
fn insufficient_imu_never_initializes() {
    // One IMU sample per frame over 3 frames = 3 total, far below imu_init_num.
    let cloud = room_cloud();
    let mut builder = MapBuilder::new(Config::default());
    for frame_index in 0..3 {
        let frame_start = frame_index as f64 * FRAME_PERIOD_SECONDS;
        let mut package = SyncPackage {
            imus: vec![IMUData {
                acc: V3D::new(0.0, 0.0, 9.81),
                gyro: V3D::zeros(),
                time: frame_start,
            }],
            cloud: cloud.clone(),
            cloud_start_time: frame_start,
            cloud_end_time: frame_start + 0.09,
        };
        builder.process(&mut package);
    }
    assert_eq!(
        builder.status(),
        BuilderStatus::ImuInit,
        "builder should stay in ImuInit without enough IMU samples"
    );
}

#[test]
fn velocity_cap_keeps_odom_under_1000() {
    // A violent single-frame IMU acceleration glitch would otherwise send the
    // estimated velocity to thousands of m/s.
    let glitch_frame = 12usize;
    let acceleration_glitch = |frame_index: usize| {
        if frame_index == glitch_frame {
            V3D::new(1.0e5, 0.0, 0.0)
        } else {
            V3D::zeros()
        }
    };

    // With the 3.1 m/s cap, reported odom must never exceed 1000 m/s.
    let mut capped_builder = MapBuilder::new(Config { max_velocity: 3.1, ..Config::default() });
    let capped_max_speed = run(&mut capped_builder, &make_frames(40, acceleration_glitch))
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);
    assert!(capped_max_speed < 1000.0, "guardrail failed: odom hit {capped_max_speed} m/s");

    // Sanity / anti-vacuity: the same glitch with the guardrail disabled really
    // does blow past 1000 m/s, proving the cap is what keeps it bounded.
    let mut uncapped_builder = MapBuilder::new(Config { max_velocity: 0.0, ..Config::default() });
    let uncapped_max_speed = run(&mut uncapped_builder, &make_frames(40, acceleration_glitch))
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);
    assert!(
        uncapped_max_speed > 1000.0,
        "expected uncapped run to diverge past 1000 m/s, got {uncapped_max_speed} (test would be vacuous)"
    );
}

#[test]
fn mapper_recovers_after_velocity_spike() {
    let glitch_frame = 12usize;
    let frames = make_frames(40, |frame_index| {
        if frame_index == glitch_frame {
            V3D::new(1.0e5, 0.0, 0.0)
        } else {
            V3D::zeros()
        }
    });

    let mut builder = MapBuilder::new(Config { max_velocity: 3.1, ..Config::default() });
    run(&mut builder, &frames);

    // After the spike is rejected and rolled back, many normal frames follow;
    // the filter should be back near the origin moving slowly, not stuck on a
    // corrupted state.
    let recovered_speed = builder.kf.x.v.norm();
    let recovered_distance = builder.kf.x.t_wi.norm();
    assert!(recovered_speed < 1.0, "velocity did not recover: {recovered_speed} m/s");
    assert!(recovered_distance < 5.0, "position did not recover: {recovered_distance} m");
}
