use nalgebra::{Matrix3, MatrixXx3, Vector3, Vector4, DVector};
use serde::{Deserialize, Deserializer};

pub type M3D = Matrix3<f64>;
pub type V3D = Vector3<f64>;
pub type V4D = Vector4<f64>;

#[derive(Clone, Copy, Debug, Default)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub curvature: f32,
}

pub type PointCloud = Vec<Point>;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    #[serde(default = "default_lidar_filter_num")]
    pub lidar_filter_num: i32,
    #[serde(default = "default_lidar_min_range")]
    pub lidar_min_range: f64,
    #[serde(default = "default_lidar_max_range")]
    pub lidar_max_range: f64,
    #[serde(default = "default_scan_resolution")]
    pub scan_resolution: f64,
    #[serde(default = "default_map_resolution")]
    pub map_resolution: f64,
    #[serde(default = "default_cube_len")]
    pub cube_len: f64,
    #[serde(default = "default_det_range")]
    pub det_range: f64,
    #[serde(default = "default_move_thresh")]
    pub move_thresh: f64,
    #[serde(default = "default_na")]
    pub na: f64,
    #[serde(default = "default_ng")]
    pub ng: f64,
    #[serde(default = "default_nba")]
    pub nba: f64,
    #[serde(default = "default_nbg")]
    pub nbg: f64,
    #[serde(default = "default_imu_init_num")]
    pub imu_init_num: usize,
    #[serde(default = "default_near_search_num")]
    pub near_search_num: usize,
    #[serde(default = "default_ieskf_max_iter")]
    pub ieskf_max_iter: usize,
    #[serde(default = "default_gravity_align")]
    pub gravity_align: bool,
    #[serde(default)]
    pub esti_il: bool,
    #[serde(default = "default_r_il", deserialize_with = "deserialize_m3d")]
    pub r_il: M3D,
    #[serde(default = "default_t_il", deserialize_with = "deserialize_v3d")]
    pub t_il: V3D,
    #[serde(default = "default_lidar_cov_inv")]
    pub lidar_cov_inv: f64,
}

fn default_r_il() -> M3D { M3D::identity() }
fn default_t_il() -> V3D { V3D::zeros() }

fn deserialize_m3d<'de, D: Deserializer<'de>>(deserializer: D) -> Result<M3D, D::Error> {
    let v: Vec<f64> = Vec::deserialize(deserializer)?;
    if v.len() == 9 {
        Ok(M3D::from_row_slice(&v))
    } else {
        Ok(M3D::identity())
    }
}

fn deserialize_v3d<'de, D: Deserializer<'de>>(deserializer: D) -> Result<V3D, D::Error> {
    let v: Vec<f64> = Vec::deserialize(deserializer)?;
    if v.len() == 3 {
        Ok(V3D::new(v[0], v[1], v[2]))
    } else {
        Ok(V3D::zeros())
    }
}

fn default_lidar_filter_num() -> i32 { 3 }
fn default_lidar_min_range() -> f64 { 0.5 }
fn default_lidar_max_range() -> f64 { 20.0 }
fn default_scan_resolution() -> f64 { 0.15 }
fn default_map_resolution() -> f64 { 0.3 }
fn default_cube_len() -> f64 { 300.0 }
fn default_det_range() -> f64 { 60.0 }
fn default_move_thresh() -> f64 { 1.5 }
fn default_na() -> f64 { 0.01 }
fn default_ng() -> f64 { 0.01 }
fn default_nba() -> f64 { 0.0001 }
fn default_nbg() -> f64 { 0.0001 }
fn default_imu_init_num() -> usize { 20 }
fn default_near_search_num() -> usize { 5 }
fn default_ieskf_max_iter() -> usize { 5 }
fn default_gravity_align() -> bool { true }
fn default_lidar_cov_inv() -> f64 { 1000.0 }

impl Default for Config {
    fn default() -> Self {
        Config {
            lidar_filter_num: 3,
            lidar_min_range: 0.5,
            lidar_max_range: 20.0,
            scan_resolution: 0.15,
            map_resolution: 0.3,
            cube_len: 300.0,
            det_range: 60.0,
            move_thresh: 1.5,
            na: 0.01,
            ng: 0.01,
            nba: 0.0001,
            nbg: 0.0001,
            imu_init_num: 20,
            near_search_num: 5,
            ieskf_max_iter: 5,
            gravity_align: true,
            esti_il: false,
            r_il: M3D::identity(),
            t_il: V3D::zeros(),
            lidar_cov_inv: 1000.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IMUData {
    pub acc: V3D,
    pub gyro: V3D,
    pub time: f64,
}

#[derive(Clone, Debug)]
pub struct Pose {
    pub offset: f64,
    pub acc: V3D,
    pub gyro: V3D,
    pub vel: V3D,
    pub trans: V3D,
    pub rot: M3D,
}

pub struct SyncPackage {
    pub imus: Vec<IMUData>,
    pub cloud: PointCloud,
    pub cloud_start_time: f64,
    pub cloud_end_time: f64,
}

pub fn esti_plane(points: &[Point], thresh: f64) -> Option<V4D> {
    let n = points.len();
    let mut a = MatrixXx3::<f64>::zeros(n);
    let mut b = DVector::<f64>::from_element(n, -1.0);

    for (i, p) in points.iter().enumerate() {
        a[(i, 0)] = p.x as f64;
        a[(i, 1)] = p.y as f64;
        a[(i, 2)] = p.z as f64;
    }

    let ata = a.transpose() * &a;
    let atb = a.transpose() * &b;
    let normvec = ata.try_inverse()? * atb;
    let norm = normvec.norm();
    let nx = normvec[0] / norm;
    let ny = normvec[1] / norm;
    let nz = normvec[2] / norm;
    let d = 1.0 / norm;

    for p in points {
        if (nx * p.x as f64 + ny * p.y as f64 + nz * p.z as f64 + d).abs() > thresh {
            return None;
        }
    }

    Some(V4D::new(nx, ny, nz, d))
}

pub fn sq_dist(p1: &Point, p2: &Point) -> f32 {
    (p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2) + (p1.z - p2.z).powi(2)
}
