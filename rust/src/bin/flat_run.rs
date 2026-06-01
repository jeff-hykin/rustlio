// Rust flat-runner: drives fastlio_rs's MapBuilder on the SAME flat sensor dump
// the C++ reference harness reads (faithful_check/dump_flat.py), using the SAME
// syncPackage logic (IMU up to cloud_end_time). This makes the Rust vs C++
// comparison byte-identical in input and identical in framing, so any state
// divergence is purely the algorithm port. Logs per-frame state as a 13-col npy
// matching ref_run's CSV columns: t,x,y,z,vx,vy,vz,gx,gy,gz,bax,bay,baz.
use std::io::Read;
use fastlio_rs::commons::*;
use fastlio_rs::map_builder::{BuilderStatus, MapBuilder};
use ndarray::Array2;
use ndarray_npy::write_npy;

struct Frame {
    t_start: f64,
    cloud: PointCloud, // points with x,y,z,intensity,curvature(ms)
}

fn read_flat(path: &str) -> (Vec<IMUData>, Vec<Frame>) {
    let mut buf = Vec::new();
    std::fs::File::open(path).expect("open flat").read_to_end(&mut buf).unwrap();
    assert_eq!(&buf[0..4], b"FLT1", "bad magic");
    let mut o = 4usize;
    let rd_u64 = |b: &[u8], o: &mut usize| { let v = u64::from_le_bytes(b[*o..*o+8].try_into().unwrap()); *o+=8; v };
    let rd_f64 = |b: &[u8], o: &mut usize| { let v = f64::from_le_bytes(b[*o..*o+8].try_into().unwrap()); *o+=8; v };
    let rd_f32 = |b: &[u8], o: &mut usize| { let v = f32::from_le_bytes(b[*o..*o+4].try_into().unwrap()); *o+=4; v };
    let n_imu = rd_u64(&buf, &mut o);
    let n_frames = rd_u64(&buf, &mut o);
    let mut imus = Vec::with_capacity(n_imu as usize);
    for _ in 0..n_imu {
        let t = rd_f64(&buf, &mut o);
        let ax = rd_f64(&buf, &mut o); let ay = rd_f64(&buf, &mut o); let az = rd_f64(&buf, &mut o);
        let gx = rd_f64(&buf, &mut o); let gy = rd_f64(&buf, &mut o); let gz = rd_f64(&buf, &mut o);
        imus.push(IMUData { acc: V3D::new(ax, ay, az), gyro: V3D::new(gx, gy, gz), time: t });
    }
    let mut frames = Vec::with_capacity(n_frames as usize);
    for _ in 0..n_frames {
        let t_start = rd_f64(&buf, &mut o);
        let np = rd_u64(&buf, &mut o);
        let mut cloud = Vec::with_capacity(np as usize);
        for _ in 0..np {
            let x = rd_f32(&buf, &mut o); let y = rd_f32(&buf, &mut o); let z = rd_f32(&buf, &mut o);
            let intensity = rd_f32(&buf, &mut o); let curvature = rd_f32(&buf, &mut o);
            cloud.push(Point { x, y, z, intensity, curvature });
        }
        frames.push(Frame { t_start, cloud });
    }
    (imus, frames)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flat = &args[1];
    let out = &args[2];
    let (imus, frames) = read_flat(flat);
    eprintln!("loaded {} imu, {} frames", imus.len(), frames.len());

    // Effective config = config_examples/mid360.yaml resolved by fastlio_rs,
    // matching faithful_check/cpp_ref/ref_run.cpp exactly.
    let mut config = Config::default();
    config.na = 0.1;
    config.ng = 0.1;
    config.det_range = 100.0;
    config.lidar_to_imu_trans = V3D::new(-0.011, -0.02329, 0.04412);
    let mut builder = MapBuilder::new(config);

    let mut imu_idx = 0usize;
    let mut recs: Vec<[f64; 13]> = Vec::new();
    for fr in &frames {
        // syncPackage: sort cloud by curvature ascending, cloud_end = start + back/1000.
        let mut cloud = fr.cloud.clone();
        cloud.sort_by(|a, b| a.curvature.partial_cmp(&b.curvature).unwrap());
        if cloud.is_empty() { continue; }
        let cloud_end = fr.t_start + cloud.last().unwrap().curvature as f64 / 1000.0;
        let mut pkg_imus = Vec::new();
        while imu_idx < imus.len() && imus[imu_idx].time < cloud_end {
            pkg_imus.push(imus[imu_idx].clone());
            imu_idx += 1;
        }
        if pkg_imus.is_empty() { continue; }
        let mut package = SyncPackage {
            imus: pkg_imus,
            cloud,
            cloud_start_time: fr.t_start,
            cloud_end_time: cloud_end,
        };
        builder.process(&mut package);
        if builder.status() == BuilderStatus::Mapping {
            let x = &builder.kf.x;
            recs.push([
                fr.t_start,
                x.imu_to_world_trans[0], x.imu_to_world_trans[1], x.imu_to_world_trans[2],
                x.v[0], x.v[1], x.v[2],
                x.g[0], x.g[1], x.g[2],
                x.ba[0], x.ba[1], x.ba[2],
            ]);
        }
    }
    let mut data = Array2::<f64>::zeros((recs.len(), 13));
    for (i, r) in recs.iter().enumerate() { for j in 0..13 { data[[i, j]] = r[j]; } }
    write_npy(out, &data).expect("write npy");
    eprintln!("wrote {} mapping frames to {}", recs.len(), out);
}
