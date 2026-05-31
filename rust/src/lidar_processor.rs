use nalgebra::{SMatrix, Vector3};
use crate::commons::*;
use crate::ieskf::{IESKF, SharedState, State, V21D};
use crate::ikd_tree::{BoxPointType, KdTree};
use crate::so3;
use crate::voxel_grid;

struct LocalMap {
    initialized: bool,
    local_map_corner: BoxPointType,
    cub_to_rm: Vec<BoxPointType>,
}

impl Default for LocalMap {
    fn default() -> Self {
        LocalMap {
            initialized: false,
            local_map_corner: BoxPointType::default(),
            cub_to_rm: Vec::new(),
        }
    }
}

pub struct LidarProcessor {
    config: Config,
    local_map: LocalMap,
    ikdtree: KdTree,
    cloud_down_lidar: PointCloud,
    cloud_down_world: PointCloud,
    point_selected_flag: Vec<bool>,
    norm_vec: PointCloud,
    nearest_points: Vec<Vec<Point>>,
}

impl LidarProcessor {
    pub fn new(config: &Config) -> Self {
        let mut ikdtree = KdTree::new(0.5, 0.6, 0.2);
        ikdtree.set_downsample_param(config.map_resolution as f32);

        LidarProcessor {
            config: config.clone(),
            local_map: LocalMap::default(),
            ikdtree,
            cloud_down_lidar: Vec::new(),
            cloud_down_world: vec![Point::default(); 10000],
            point_selected_flag: vec![false; 10000],
            norm_vec: vec![Point::default(); 10000],
            nearest_points: vec![Vec::new(); 10000],
        }
    }

    pub fn r_wl(&self, kf: &IESKF) -> M3D {
        kf.x.r_wi * kf.x.r_il
    }

    pub fn t_wl(&self, kf: &IESKF) -> V3D {
        kf.x.t_wi + kf.x.r_wi * kf.x.t_il
    }

    pub fn init_cloud_map(&mut self, points: &[Point]) {
        self.ikdtree.build(points.to_vec());
    }

    pub fn process(&mut self, package: &SyncPackage, kf: &mut IESKF) {
        if self.config.scan_resolution > 0.0 {
            self.cloud_down_lidar =
                voxel_grid::downsample(&package.cloud, self.config.scan_resolution);
        } else {
            self.cloud_down_lidar = package.cloud.clone();
        }

        self.trim_cloud_map(kf);
        self.run_update(kf);
        self.incr_cloud_map(kf);
    }

    fn run_update(&mut self, kf: &mut IESKF) {
        let cloud_down_lidar = self.cloud_down_lidar.clone();
        let config = self.config.clone();
        let size = cloud_down_lidar.len();

        let mut cloud_dw = vec![Point::default(); size];
        let mut psf = vec![false; size];
        let mut nv = vec![Point::default(); size];
        let mut np: Vec<Vec<Point>> = vec![Vec::new(); size];

        {
            let ikdtree = &mut self.ikdtree;
            let mut loss_func = move |state: &State, share_data: &mut SharedState| {
                Self::update_loss_func_inner(
                    state,
                    share_data,
                    &cloud_down_lidar,
                    &config,
                    &mut cloud_dw,
                    &mut psf,
                    &mut nv,
                    &mut np,
                    ikdtree,
                );
            };
            let stop_func = |delta: &V21D| -> bool {
                let rot_delta: Vector3<f64> = delta.fixed_rows::<3>(0).into_owned();
                let t_delta: Vector3<f64> = delta.fixed_rows::<3>(3).into_owned();
                (rot_delta.norm() * 57.3 < 0.01) && (t_delta.norm() * 100.0 < 0.015)
            };
            kf.update(&mut loss_func, &stop_func);
        }

        let state = &kf.x;
        self.cloud_down_world = vec![Point::default(); self.cloud_down_lidar.len()];
        self.nearest_points = vec![Vec::new(); self.cloud_down_lidar.len()];
        self.point_selected_flag = vec![false; self.cloud_down_lidar.len()];
        self.norm_vec = vec![Point::default(); self.cloud_down_lidar.len()];

        for i in 0..self.cloud_down_lidar.len() {
            let p = &self.cloud_down_lidar[i];
            let pv = V3D::new(p.x as f64, p.y as f64, p.z as f64);
            let pw = state.r_wi * (state.r_il * pv + state.t_il) + state.t_wi;
            self.cloud_down_world[i] = Point {
                x: pw[0] as f32,
                y: pw[1] as f32,
                z: pw[2] as f32,
                intensity: p.intensity,
                curvature: p.curvature,
            };
            let (pts, _dists) = self.ikdtree.nearest_search(
                &self.cloud_down_world[i],
                self.config.near_search_num,
                f32::INFINITY,
            );
            self.nearest_points[i] = pts;
        }
    }

    fn update_loss_func_inner(
        state: &State,
        share_data: &mut SharedState,
        cloud_down_lidar: &[Point],
        config: &Config,
        cloud_down_world: &mut Vec<Point>,
        point_selected_flag: &mut Vec<bool>,
        norm_vec: &mut Vec<Point>,
        nearest_points: &mut Vec<Vec<Point>>,
        ikdtree: &mut KdTree,
    ) {
        let size = cloud_down_lidar.len();

        for i in 0..size {
            let pb = &cloud_down_lidar[i];
            let pbv = V3D::new(pb.x as f64, pb.y as f64, pb.z as f64);
            let pwv = state.r_wi * (state.r_il * pbv + state.t_il) + state.t_wi;

            cloud_down_world[i] = Point {
                x: pwv[0] as f32,
                y: pwv[1] as f32,
                z: pwv[2] as f32,
                intensity: pb.intensity,
                curvature: pb.curvature,
            };

            let (pts, dists) = ikdtree.nearest_search(
                &cloud_down_world[i],
                config.near_search_num,
                f32::INFINITY,
            );
            nearest_points[i] = pts;

            if nearest_points[i].len() >= config.near_search_num
                && dists.last().map_or(false, |d| *d <= 5.0)
            {
                point_selected_flag[i] = true;
            } else {
                point_selected_flag[i] = false;
            }

            if !point_selected_flag[i] {
                continue;
            }

            if let Some(pabcd) = esti_plane(&nearest_points[i], 0.1) {
                let pd2 = pabcd[0] * pwv[0] + pabcd[1] * pwv[1] + pabcd[2] * pwv[2] + pabcd[3];
                let s = 1.0 - 0.9 * pd2.abs() / pbv.norm().sqrt();
                if s > 0.9 {
                    point_selected_flag[i] = true;
                    norm_vec[i] = Point {
                        x: pabcd[0] as f32,
                        y: pabcd[1] as f32,
                        z: pabcd[2] as f32,
                        intensity: pd2 as f32,
                        curvature: 0.0,
                    };
                } else {
                    point_selected_flag[i] = false;
                }
            } else {
                point_selected_flag[i] = false;
            }
        }

        let mut effect_cloud_lidar = Vec::new();
        let mut effect_norm_vec = Vec::new();
        for i in 0..size {
            if point_selected_flag[i] {
                effect_cloud_lidar.push(cloud_down_lidar[i]);
                effect_norm_vec.push(norm_vec[i]);
            }
        }

        if effect_cloud_lidar.is_empty() {
            share_data.valid = false;
            use std::sync::atomic::{AtomicBool, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                eprintln!("NO Effective Points! (suppressing further occurrences)");
            }
            return;
        }
        share_data.valid = true;
        share_data.h = crate::ieskf::M12D::zeros();
        share_data.b = crate::ieskf::V12D::zeros();

        for i in 0..effect_cloud_lidar.len() {
            let mut j_row = SMatrix::<f64, 1, 12>::zeros();
            let lp = &effect_cloud_lidar[i];
            let np = &effect_norm_vec[i];
            let lpv = V3D::new(lp.x as f64, lp.y as f64, lp.z as f64);
            let nv = V3D::new(np.x as f64, np.y as f64, np.z as f64);

            let b_val = -nv.transpose()
                * state.r_wi
                * so3::hat(&(state.r_il * lpv + state.t_wi));
            j_row.fixed_view_mut::<1, 3>(0, 0).copy_from(&b_val);
            j_row
                .fixed_view_mut::<1, 3>(0, 3)
                .copy_from(&nv.transpose());

            if config.esti_il {
                let c_val =
                    -nv.transpose() * state.r_wi * state.r_il * so3::hat(&lpv);
                let d_val = nv.transpose() * state.r_wi;
                j_row.fixed_view_mut::<1, 3>(0, 6).copy_from(&c_val);
                j_row.fixed_view_mut::<1, 3>(0, 9).copy_from(&d_val);
            }

            share_data.h += j_row.transpose() * config.lidar_cov_inv * j_row;
            share_data.b +=
                (j_row.transpose() * config.lidar_cov_inv * np.intensity as f64)
                    .fixed_rows::<12>(0)
                    .into_owned();
        }
    }

    pub fn trim_cloud_map(&mut self, kf: &IESKF) {
        self.local_map.cub_to_rm.clear();
        let state = &kf.x;
        let pos_lidar = state.t_wi + state.r_wi * state.t_il;

        if !self.local_map.initialized {
            for i in 0..3 {
                self.local_map.local_map_corner.vertex_min[i] =
                    pos_lidar[i] as f32 - self.config.cube_len as f32 / 2.0;
                self.local_map.local_map_corner.vertex_max[i] =
                    pos_lidar[i] as f32 + self.config.cube_len as f32 / 2.0;
            }
            self.local_map.initialized = true;
            return;
        }

        let det_thresh = self.config.move_thresh * self.config.det_range;
        let mut need_move = false;
        let mut dist_to_edge = [[0.0f64; 2]; 3];
        for i in 0..3 {
            dist_to_edge[i][0] =
                (pos_lidar[i] - self.local_map.local_map_corner.vertex_min[i] as f64).abs();
            dist_to_edge[i][1] =
                (pos_lidar[i] - self.local_map.local_map_corner.vertex_max[i] as f64).abs();
            if dist_to_edge[i][0] <= det_thresh || dist_to_edge[i][1] <= det_thresh {
                need_move = true;
            }
        }
        if !need_move {
            return;
        }

        let mut new_corner = self.local_map.local_map_corner;
        let mov_dist = ((self.config.cube_len - 2.0 * self.config.move_thresh * self.config.det_range)
            * 0.5
            * 0.9)
            .max(self.config.det_range * (self.config.move_thresh - 1.0))
            as f32;

        for i in 0..3 {
            let mut temp_corner = self.local_map.local_map_corner;
            if dist_to_edge[i][0] <= det_thresh {
                new_corner.vertex_max[i] -= mov_dist;
                new_corner.vertex_min[i] -= mov_dist;
                temp_corner.vertex_min[i] =
                    self.local_map.local_map_corner.vertex_max[i] - mov_dist;
                self.local_map.cub_to_rm.push(temp_corner);
            } else if dist_to_edge[i][1] <= det_thresh {
                new_corner.vertex_max[i] += mov_dist;
                new_corner.vertex_min[i] += mov_dist;
                temp_corner.vertex_max[i] =
                    self.local_map.local_map_corner.vertex_min[i] + mov_dist;
                self.local_map.cub_to_rm.push(temp_corner);
            }
        }
        self.local_map.local_map_corner = new_corner;

        let _removed = self.ikdtree.acquire_removed_points();
        if !self.local_map.cub_to_rm.is_empty() {
            self.ikdtree
                .delete_point_boxes(&self.local_map.cub_to_rm);
        }
    }

    pub fn incr_cloud_map(&mut self, kf: &IESKF) {
        if self.cloud_down_lidar.is_empty() {
            return;
        }
        let state = &kf.x;
        let size = self.cloud_down_lidar.len();
        let mut point_to_add = Vec::new();
        let mut point_no_need_downsample = Vec::new();

        for i in 0..size {
            let p = &self.cloud_down_lidar[i];
            let pv = V3D::new(p.x as f64, p.y as f64, p.z as f64);
            let pw = state.r_wi * (state.r_il * pv + state.t_il) + state.t_wi;
            let pw_point = Point {
                x: pw[0] as f32,
                y: pw[1] as f32,
                z: pw[2] as f32,
                intensity: p.intensity,
                curvature: p.curvature,
            };

            if self.nearest_points[i].is_empty() {
                point_to_add.push(pw_point);
                continue;
            }

            let points_near = &self.nearest_points[i];
            let res = self.config.map_resolution as f32;
            let mid_point = Point {
                x: (pw_point.x / res).floor() * res + 0.5 * res,
                y: (pw_point.y / res).floor() * res + 0.5 * res,
                z: (pw_point.z / res).floor() * res + 0.5 * res,
                intensity: 0.0,
                curvature: 0.0,
            };

            if (points_near[0].x - mid_point.x).abs() > 0.5 * res
                && (points_near[0].y - mid_point.y).abs() > 0.5 * res
                && (points_near[0].z - mid_point.z).abs() > 0.5 * res
            {
                point_no_need_downsample.push(pw_point);
                continue;
            }

            let dist = sq_dist(&pw_point, &mid_point);
            let mut need_add = true;
            for j in 0..self.config.near_search_num {
                if points_near.len() < self.config.near_search_num {
                    break;
                }
                if sq_dist(&points_near[j], &mid_point) < dist {
                    need_add = false;
                    break;
                }
            }
            if need_add {
                point_to_add.push(pw_point);
            }
        }

        self.ikdtree.add_points(&point_to_add, true);
        self.ikdtree.add_points(&point_no_need_downsample, false);
    }

    pub fn transform_cloud(cloud: &[Point], r: &M3D, t: &V3D) -> PointCloud {
        cloud
            .iter()
            .map(|p| {
                let pv = V3D::new(p.x as f64, p.y as f64, p.z as f64);
                let pw = r * pv + t;
                Point {
                    x: pw[0] as f32,
                    y: pw[1] as f32,
                    z: pw[2] as f32,
                    intensity: p.intensity,
                    curvature: p.curvature,
                }
            })
            .collect()
    }
}
