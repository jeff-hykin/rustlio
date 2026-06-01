use nalgebra::{Matrix3, MatrixXx3, Vector3, Vector4, DVector};
use serde::Deserialize;

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

/// Runtime configuration consumed by `MapBuilder`.
///
/// This is a plain data struct with no YAML/serde coupling. Parse a config
/// file with [`Config::from_yaml_str`] / [`Config::from_yaml_path`], which
/// accepts both this repo's flat `lio.yaml` schema and the upstream FAST-LIO
/// nested schema (`common`/`preprocess`/`mapping`) used by `config_examples`.
#[derive(Clone, Debug)]
pub struct Config {
    pub imu_topic: String,
    pub lidar_topic: String,
    pub lidar_type: i32,
    pub lidar_filter_num: i32,
    pub lidar_min_range: f64,
    pub lidar_max_range: f64,
    pub scan_resolution: f64,
    pub map_resolution: f64,
    pub cube_len: f64,
    pub det_range: f64,
    pub move_thresh: f64,
    pub na: f64,
    pub ng: f64,
    pub nba: f64,
    pub nbg: f64,
    pub imu_init_num: usize,
    pub near_search_num: usize,
    pub ieskf_max_iter: usize,
    pub gravity_align: bool,
    pub esti_il: bool,
    pub r_il: M3D,
    pub t_il: V3D,
    pub lidar_cov_inv: f64,
    pub max_velocity: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            imu_topic: "/livox/imu".to_string(),
            lidar_topic: "/livox/lidar".to_string(),
            lidar_type: 1,
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
            max_velocity: 3.1,
        }
    }
}

impl Config {
    /// Parse a YAML string (flat or upstream-nested) into a `Config`.
    /// Missing keys fall back to [`Config::default`].
    pub fn from_yaml_str(s: &str) -> Result<Config, serde_yaml::Error> {
        let raw: RawConfig = serde_yaml::from_str(s)?;
        Ok(raw.into_config())
    }

    /// Parse a YAML file at `path` into a `Config`.
    pub fn from_yaml_path<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Config> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_yaml_str(&contents)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

fn vec_to_m3d(v: &[f64]) -> Option<M3D> {
    (v.len() == 9).then(|| M3D::from_row_slice(v))
}

fn vec_to_v3d(v: &[f64]) -> Option<V3D> {
    (v.len() == 3).then(|| V3D::new(v[0], v[1], v[2]))
}

/// Deserialization shim. Every field is optional so a partial config (in
/// either schema) merges onto `Config::default()`. Unknown keys (e.g. the
/// upstream `publish`/`pcd_save` sections, `body_frame`, `scan_line`) are
/// ignored. Where a value can be supplied by both schemas, the flat top-level
/// key wins over the nested section.
#[derive(Deserialize, Default)]
struct RawConfig {
    // Upstream FAST-LIO nested sections.
    common: Option<CommonSection>,
    preprocess: Option<PreprocessSection>,
    mapping: Option<MappingSection>,

    // Flat schema (this repo's `lio.yaml`).
    imu_topic: Option<String>,
    lidar_topic: Option<String>,
    lidar_type: Option<i32>,
    lidar_filter_num: Option<i32>,
    lidar_min_range: Option<f64>,
    lidar_max_range: Option<f64>,
    scan_resolution: Option<f64>,
    map_resolution: Option<f64>,
    cube_len: Option<f64>,
    det_range: Option<f64>,
    move_thresh: Option<f64>,
    na: Option<f64>,
    ng: Option<f64>,
    nba: Option<f64>,
    nbg: Option<f64>,
    imu_init_num: Option<usize>,
    near_search_num: Option<usize>,
    ieskf_max_iter: Option<usize>,
    gravity_align: Option<bool>,
    esti_il: Option<bool>,
    r_il: Option<Vec<f64>>,
    t_il: Option<Vec<f64>>,
    lidar_cov_inv: Option<f64>,
    max_velocity: Option<f64>,
}

#[derive(Deserialize, Default)]
struct CommonSection {
    imu_topic: Option<String>,
    lid_topic: Option<String>,
}

#[derive(Deserialize, Default)]
struct PreprocessSection {
    lidar_type: Option<i32>,
    blind: Option<f64>,
    point_filter_num: Option<i32>,
}

#[derive(Deserialize, Default)]
struct MappingSection {
    acc_cov: Option<f64>,
    gyr_cov: Option<f64>,
    b_acc_cov: Option<f64>,
    b_gyr_cov: Option<f64>,
    det_range: Option<f64>,
    cube_side_length: Option<f64>,
    filter_size_surf: Option<f64>,
    filter_size_map: Option<f64>,
    extrinsic_est_en: Option<bool>,
    #[serde(rename = "extrinsic_R")]
    extrinsic_r: Option<Vec<f64>>,
    #[serde(rename = "extrinsic_T")]
    extrinsic_t: Option<Vec<f64>>,
}

impl RawConfig {
    fn into_config(self) -> Config {
        let mut c = Config::default();
        let common = self.common.unwrap_or_default();
        let pre = self.preprocess.unwrap_or_default();
        let map = self.mapping.unwrap_or_default();

        if let Some(v) = self.imu_topic.or(common.imu_topic) { c.imu_topic = v; }
        if let Some(v) = self.lidar_topic.or(common.lid_topic) { c.lidar_topic = v; }
        if let Some(v) = self.lidar_type.or(pre.lidar_type) { c.lidar_type = v; }
        if let Some(v) = self.lidar_filter_num.or(pre.point_filter_num) { c.lidar_filter_num = v; }
        if let Some(v) = self.lidar_min_range.or(pre.blind) { c.lidar_min_range = v; }
        if let Some(v) = self.lidar_max_range { c.lidar_max_range = v; }
        if let Some(v) = self.scan_resolution.or(map.filter_size_surf) { c.scan_resolution = v; }
        if let Some(v) = self.map_resolution.or(map.filter_size_map) { c.map_resolution = v; }
        if let Some(v) = self.cube_len.or(map.cube_side_length) { c.cube_len = v; }
        if let Some(v) = self.det_range.or(map.det_range) { c.det_range = v; }
        if let Some(v) = self.move_thresh { c.move_thresh = v; }
        if let Some(v) = self.na.or(map.acc_cov) { c.na = v; }
        if let Some(v) = self.ng.or(map.gyr_cov) { c.ng = v; }
        if let Some(v) = self.nba.or(map.b_acc_cov) { c.nba = v; }
        if let Some(v) = self.nbg.or(map.b_gyr_cov) { c.nbg = v; }
        if let Some(v) = self.imu_init_num { c.imu_init_num = v; }
        if let Some(v) = self.near_search_num { c.near_search_num = v; }
        if let Some(v) = self.ieskf_max_iter { c.ieskf_max_iter = v; }
        if let Some(v) = self.gravity_align { c.gravity_align = v; }
        if let Some(v) = self.esti_il.or(map.extrinsic_est_en) { c.esti_il = v; }
        if let Some(v) = self.r_il.or(map.extrinsic_r).as_deref().and_then(vec_to_m3d) { c.r_il = v; }
        if let Some(v) = self.t_il.or(map.extrinsic_t).as_deref().and_then(vec_to_v3d) { c.t_il = v; }
        if let Some(v) = self.lidar_cov_inv { c.lidar_cov_inv = v; }
        if let Some(v) = self.max_velocity { c.max_velocity = v; }
        c
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
