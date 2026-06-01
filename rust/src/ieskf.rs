use nalgebra::{SMatrix, SVector, Matrix3, Vector3};
use crate::so3;
use crate::commons::{M3D, V3D};

pub type M12D = SMatrix<f64, 12, 12>;
pub type V12D = SVector<f64, 12>;
/// Error-state dimension: rot, pos, lidar_to_imu_rot, lidar_to_imu_trans, v, bg, ba, gravity = 8 * 3 = 24.
/// Gravity (rows 21..24) is estimated online as a 3-DOF additive perturbation.
pub type M24D = SMatrix<f64, 24, 24>;
pub type V24D = SVector<f64, 24>;
pub type M24X12D = SMatrix<f64, 24, 12>;

// Back-compat aliases (the error-state grew from 21 to 24 dims when online
// gravity estimation was added).
pub type M21D = M24D;
pub type V21D = V24D;
pub type M21X12D = M24X12D;

pub struct SharedState {
    pub h: M12D,
    pub b: V12D,
    pub res: f64,
    pub valid: bool,
    pub iter_num: usize,
}

impl Default for SharedState {
    fn default() -> Self {
        SharedState {
            h: M12D::zeros(),
            b: V12D::zeros(),
            res: 1e10,
            valid: false,
            iter_num: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Input {
    pub acc: V3D,
    pub gyro: V3D,
}

impl Default for Input {
    fn default() -> Self {
        Input {
            acc: V3D::zeros(),
            gyro: V3D::zeros(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct State {
    pub imu_to_world_rot: M3D,
    pub imu_to_world_trans: V3D,
    pub lidar_to_imu_rot: M3D,
    pub lidar_to_imu_trans: V3D,
    pub v: V3D,
    pub bg: V3D,
    pub ba: V3D,
    pub g: V3D,
}

impl State {
    pub const GRAVITY: f64 = 9.81;

    pub fn init_gravity_dir(&mut self, gravity_dir: &V3D) {
        self.g = gravity_dir.normalize() * Self::GRAVITY;
    }

    pub fn add_delta(&mut self, delta: &V21D) {
        self.imu_to_world_rot *= so3::exp(&delta.fixed_rows::<3>(0).into_owned());
        self.imu_to_world_trans += delta.fixed_rows::<3>(3).into_owned();
        self.lidar_to_imu_rot *= so3::exp(&delta.fixed_rows::<3>(6).into_owned());
        self.lidar_to_imu_trans += delta.fixed_rows::<3>(9).into_owned();
        self.v += delta.fixed_rows::<3>(12).into_owned();
        self.bg += delta.fixed_rows::<3>(15).into_owned();
        self.ba += delta.fixed_rows::<3>(18).into_owned();
        self.g += delta.fixed_rows::<3>(21).into_owned();
    }

    pub fn minus(&self, other: &State) -> V21D {
        let mut delta = V21D::zeros();
        delta.fixed_rows_mut::<3>(0).copy_from(
            &so3::log(&(other.imu_to_world_rot.transpose() * self.imu_to_world_rot)),
        );
        delta
            .fixed_rows_mut::<3>(3)
            .copy_from(&(self.imu_to_world_trans - other.imu_to_world_trans));
        delta.fixed_rows_mut::<3>(6).copy_from(
            &so3::log(&(other.lidar_to_imu_rot.transpose() * self.lidar_to_imu_rot)),
        );
        delta
            .fixed_rows_mut::<3>(9)
            .copy_from(&(self.lidar_to_imu_trans - other.lidar_to_imu_trans));
        delta
            .fixed_rows_mut::<3>(12)
            .copy_from(&(self.v - other.v));
        delta
            .fixed_rows_mut::<3>(15)
            .copy_from(&(self.bg - other.bg));
        delta
            .fixed_rows_mut::<3>(18)
            .copy_from(&(self.ba - other.ba));
        delta
            .fixed_rows_mut::<3>(21)
            .copy_from(&(self.g - other.g));
        delta
    }
}

impl Default for State {
    fn default() -> Self {
        State {
            imu_to_world_rot: M3D::identity(),
            imu_to_world_trans: V3D::zeros(),
            lidar_to_imu_rot: M3D::identity(),
            lidar_to_imu_trans: V3D::zeros(),
            v: V3D::zeros(),
            bg: V3D::zeros(),
            ba: V3D::zeros(),
            g: V3D::new(0.0, 0.0, -9.81),
        }
    }
}

pub struct IESKF {
    max_iter: usize,
    pub x: State,
    pub p: M21D,
    f_mat: M21D,
    g_mat: M21X12D,
}

impl IESKF {
    pub fn new() -> Self {
        IESKF {
            max_iter: 10,
            x: State::default(),
            p: M21D::identity(),
            f_mat: M21D::identity(),
            g_mat: M21X12D::zeros(),
        }
    }

    pub fn set_max_iter(&mut self, iter: usize) {
        self.max_iter = iter;
    }

    pub fn predict(&mut self, inp: &Input, dt: f64, q: &M12D) {
        let mut delta = V21D::zeros();
        let gyro_corrected = inp.gyro - self.x.bg;
        let acc_corrected = inp.acc - self.x.ba;

        delta
            .fixed_rows_mut::<3>(0)
            .copy_from(&(gyro_corrected * dt));
        delta.fixed_rows_mut::<3>(3).copy_from(&(self.x.v * dt));
        delta
            .fixed_rows_mut::<3>(12)
            .copy_from(&((self.x.imu_to_world_rot * acc_corrected + self.x.g) * dt));

        self.f_mat = M21D::identity();
        let neg_gyro_dt = -(gyro_corrected * dt);
        self.f_mat
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&so3::exp(&neg_gyro_dt));
        let jr_val = so3::jr(&(gyro_corrected * dt));
        self.f_mat
            .fixed_view_mut::<3, 3>(0, 15)
            .copy_from(&(-jr_val * dt));
        self.f_mat
            .fixed_view_mut::<3, 3>(3, 12)
            .copy_from(&(Matrix3::identity() * dt));
        self.f_mat
            .fixed_view_mut::<3, 3>(12, 0)
            .copy_from(&(-self.x.imu_to_world_rot * so3::hat(&acc_corrected) * dt));
        self.f_mat
            .fixed_view_mut::<3, 3>(12, 18)
            .copy_from(&(-self.x.imu_to_world_rot * dt));
        // velocity depends on gravity: d(delta_v)/d(delta_g) = I * dt
        self.f_mat
            .fixed_view_mut::<3, 3>(12, 21)
            .copy_from(&(Matrix3::identity() * dt));

        self.g_mat = M21X12D::zeros();
        self.g_mat
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-jr_val * dt));
        self.g_mat
            .fixed_view_mut::<3, 3>(12, 3)
            .copy_from(&(-self.x.imu_to_world_rot * dt));
        self.g_mat
            .fixed_view_mut::<3, 3>(15, 6)
            .copy_from(&(Matrix3::identity() * dt));
        self.g_mat
            .fixed_view_mut::<3, 3>(18, 9)
            .copy_from(&(Matrix3::identity() * dt));

        self.x.add_delta(&delta);
        self.p = self.f_mat * self.p * self.f_mat.transpose()
            + self.g_mat * q * self.g_mat.transpose();
    }

    pub fn update(
        &mut self,
        loss_func: &mut dyn FnMut(&State, &mut SharedState),
        stop_func: &dyn Fn(&V21D) -> bool,
    ) {
        let predict_x = self.x.clone();
        let mut shared_data = SharedState::default();

        for _i in 0..self.max_iter {
            loss_func(&self.x, &mut shared_data);
            if !shared_data.valid {
                break;
            }

            let mut h = M21D::zeros();
            let mut b = V21D::zeros();
            let delta = self.x.minus(&predict_x);

            let mut j = M21D::identity();
            j.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&so3::jr_inv(&delta.fixed_rows::<3>(0).into_owned()));
            j.fixed_view_mut::<3, 3>(6, 6)
                .copy_from(&so3::jr_inv(&delta.fixed_rows::<3>(6).into_owned()));

            let p_inv = self.p.try_inverse().unwrap_or(M21D::identity());
            h += j.transpose() * p_inv * j;
            b += j.transpose() * p_inv * delta;

            let h_block = h.fixed_view::<12, 12>(0, 0).into_owned() + shared_data.h;
            h.fixed_view_mut::<12, 12>(0, 0).copy_from(&h_block);
            let b_block = b.fixed_rows::<12>(0).into_owned() + shared_data.b;
            b.fixed_rows_mut::<12>(0).copy_from(&b_block);

            let delta = -(h.try_inverse().unwrap_or(M21D::identity())) * b;
            self.x.add_delta(&delta);
            shared_data.iter_num += 1;

            if stop_func(&delta) {
                break;
            }
        }

        let delta = self.x.minus(&predict_x);
        let mut l = M21D::identity();
        l.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&so3::jr(&delta.fixed_rows::<3>(0).into_owned()));
        l.fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&so3::jr(&delta.fixed_rows::<3>(6).into_owned()));

        let h_full = {
            let mut h = M21D::zeros();
            let mut j = M21D::identity();
            j.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&so3::jr_inv(&delta.fixed_rows::<3>(0).into_owned()));
            j.fixed_view_mut::<3, 3>(6, 6)
                .copy_from(&so3::jr_inv(&delta.fixed_rows::<3>(6).into_owned()));
            let p_inv = self.p.try_inverse().unwrap_or(M21D::identity());
            h += j.transpose() * p_inv * j;
            let h_block = h.fixed_view::<12, 12>(0, 0).into_owned() + shared_data.h;
            h.fixed_view_mut::<12, 12>(0, 0).copy_from(&h_block);
            h
        };

        self.p =
            l * h_full.try_inverse().unwrap_or(M21D::identity()) * l.transpose();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_advances_state() {
        let mut kf = IESKF::new();
        let q = M12D::identity() * 0.01;
        let inp = Input {
            acc: V3D::new(1.0, 0.0, 9.81),
            gyro: V3D::zeros(),
        };
        let pos_before = kf.x.imu_to_world_trans;
        kf.predict(&inp, 0.01, &q);
        assert_ne!(kf.x.v, V3D::zeros());
        kf.predict(&inp, 0.01, &q);
        assert_ne!(kf.x.imu_to_world_trans, pos_before);
    }
}
