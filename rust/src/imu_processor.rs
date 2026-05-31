use crate::commons::*;
use crate::ieskf::{IESKF, Input, M12D};
use crate::so3;
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

pub struct IMUProcessor {
    config: Config,
    last_propagate_end_time: f64,
    imu_cache: Vec<IMUData>,
    poses_cache: Vec<Pose>,
    last_acc: V3D,
    last_gyro: V3D,
    q: M12D,
    last_imu: Option<IMUData>,
}

impl IMUProcessor {
    pub fn new(config: &Config) -> Self {
        let mut q = M12D::identity();
        q.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(M3D::identity() * config.ng));
        q.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(M3D::identity() * config.na));
        q.fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&(M3D::identity() * config.nbg));
        q.fixed_view_mut::<3, 3>(9, 9)
            .copy_from(&(M3D::identity() * config.nba));

        IMUProcessor {
            config: config.clone(),
            last_propagate_end_time: 0.0,
            imu_cache: Vec::new(),
            poses_cache: Vec::new(),
            last_acc: V3D::zeros(),
            last_gyro: V3D::zeros(),
            q,
            last_imu: None,
        }
    }

    pub fn initialize(&mut self, package: &SyncPackage, kf: &mut IESKF) -> bool {
        self.imu_cache.extend(package.imus.iter().cloned());
        if self.imu_cache.len() < self.config.imu_init_num {
            return false;
        }

        let mut acc_mean = V3D::zeros();
        let mut gyro_mean = V3D::zeros();
        let n = self.imu_cache.len() as f64;
        for imu in &self.imu_cache {
            acc_mean += imu.acc;
            gyro_mean += imu.gyro;
        }
        acc_mean /= n;
        gyro_mean /= n;

        kf.x.r_il = self.config.r_il;
        kf.x.t_il = self.config.t_il;
        kf.x.bg = gyro_mean;

        if self.config.gravity_align {
            let neg_acc = -acc_mean;
            let from = neg_acc.normalize();
            let to = V3D::new(0.0, 0.0, -1.0);
            let q = UnitQuaternion::rotation_between(&from, &to)
                .unwrap_or(UnitQuaternion::identity());
            kf.x.r_wi = *q.to_rotation_matrix().matrix();
            kf.x.init_gravity_dir(&V3D::new(0.0, 0.0, -1.0));
        } else {
            kf.x.init_gravity_dir(&(-acc_mean));
        }

        kf.p = nalgebra::SMatrix::<f64, 21, 21>::identity();
        kf.p.fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&(M3D::identity() * 0.00001));
        kf.p.fixed_view_mut::<3, 3>(9, 9)
            .copy_from(&(M3D::identity() * 0.00001));
        kf.p.fixed_view_mut::<3, 3>(15, 15)
            .copy_from(&(M3D::identity() * 0.0001));
        kf.p.fixed_view_mut::<3, 3>(18, 18)
            .copy_from(&(M3D::identity() * 0.0001));

        self.last_imu = self.imu_cache.last().cloned();
        self.last_propagate_end_time = package.cloud_end_time;
        true
    }

    pub fn undistort(&mut self, package: &mut SyncPackage, kf: &mut IESKF) {
        self.imu_cache.clear();
        if let Some(ref last) = self.last_imu {
            self.imu_cache.push(last.clone());
        }
        self.imu_cache.extend(package.imus.iter().cloned());

        let imu_time_end = self.imu_cache.last().map(|i| i.time).unwrap_or(0.0);
        let cloud_time_begin = package.cloud_start_time;
        let propagate_time_end = package.cloud_end_time;

        self.poses_cache.clear();
        self.poses_cache.push(Pose {
            offset: 0.0,
            acc: self.last_acc,
            gyro: self.last_gyro,
            vel: kf.x.v,
            trans: kf.x.t_wi,
            rot: kf.x.r_wi,
        });

        let mut inp = Input::default();
        if let Some(last) = self.imu_cache.last() {
            inp.acc = last.acc;
            inp.gyro = last.gyro;
        }

        for i in 0..self.imu_cache.len().saturating_sub(1) {
            let head = &self.imu_cache[i];
            let tail = &self.imu_cache[i + 1];

            if tail.time < self.last_propagate_end_time {
                continue;
            }

            let gyro_val = 0.5 * (head.gyro + tail.gyro);
            let acc_val = 0.5 * (head.acc + tail.acc);

            let dt = if head.time < self.last_propagate_end_time {
                tail.time - self.last_propagate_end_time
            } else {
                tail.time - head.time
            };

            inp.acc = acc_val;
            inp.gyro = gyro_val;
            kf.predict(&inp, dt, &self.q);

            self.last_gyro = gyro_val - kf.x.bg;
            self.last_acc = kf.x.r_wi * (acc_val - kf.x.ba) + kf.x.g;

            let offset = tail.time - cloud_time_begin;
            self.poses_cache.push(Pose {
                offset,
                acc: self.last_acc,
                gyro: self.last_gyro,
                vel: kf.x.v,
                trans: kf.x.t_wi,
                rot: kf.x.r_wi,
            });
        }

        let dt = propagate_time_end - imu_time_end;
        kf.predict(&inp, dt, &self.q);
        self.last_imu = self.imu_cache.last().cloned();
        self.last_propagate_end_time = propagate_time_end;

        let cur_r_wi = kf.x.r_wi;
        let cur_t_wi = kf.x.t_wi;
        let cur_r_il = kf.x.r_il;
        let cur_t_il = kf.x.t_il;

        let n_points = package.cloud.len();
        if n_points == 0 || self.poses_cache.len() < 2 {
            return;
        }

        let mut pcl_idx = n_points - 1;
        for kp_idx in (1..self.poses_cache.len()).rev() {
            let head = &self.poses_cache[kp_idx - 1];
            let tail = &self.poses_cache[kp_idx];

            let imu_r_wi = head.rot;
            let imu_t_wi = head.trans;
            let imu_vel = head.vel;
            let imu_acc = tail.acc;
            let imu_gyro = tail.gyro;

            loop {
                let pt_offset = package.cloud[pcl_idx].curvature as f64 / 1000.0;
                if pt_offset <= head.offset {
                    break;
                }

                let dt = pt_offset - head.offset;
                let p = &package.cloud[pcl_idx];
                let point = V3D::new(p.x as f64, p.y as f64, p.z as f64);

                let point_rot = imu_r_wi * so3::exp(&(imu_gyro * dt));
                let point_pos = imu_t_wi + imu_vel * dt + 0.5 * imu_acc * dt * dt;

                let p_compensate = cur_r_il.transpose()
                    * (cur_r_wi.transpose()
                        * (point_rot * (cur_r_il * point + cur_t_il) + point_pos
                            - cur_t_wi)
                        - cur_t_il);

                package.cloud[pcl_idx].x = p_compensate[0] as f32;
                package.cloud[pcl_idx].y = p_compensate[1] as f32;
                package.cloud[pcl_idx].z = p_compensate[2] as f32;

                if pcl_idx == 0 {
                    break;
                }
                pcl_idx -= 1;
            }
            if pcl_idx == 0 {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ieskf::IESKF;

    #[test]
    fn test_initialization_convergence() {
        let config = Config::default();
        let mut proc = IMUProcessor::new(&config);
        let mut kf = IESKF::new();

        let mut imus = Vec::new();
        for i in 0..config.imu_init_num {
            imus.push(IMUData {
                acc: V3D::new(0.0, 0.0, 9.81),
                gyro: V3D::new(0.001, -0.002, 0.0005),
                time: i as f64 * 0.01,
            });
        }
        let end_time = (config.imu_init_num - 1) as f64 * 0.01;

        let package = SyncPackage {
            imus,
            cloud: vec![],
            cloud_start_time: 0.0,
            cloud_end_time: end_time,
        };

        let converged = proc.initialize(&package, &mut kf);
        assert!(converged);

        let g_mag = kf.x.g.norm();
        assert!((g_mag - 9.81).abs() < 0.1, "gravity magnitude: {}", g_mag);

        assert!(kf.x.bg.norm() < 0.01, "gyro bias: {:?}", kf.x.bg);
    }
}
