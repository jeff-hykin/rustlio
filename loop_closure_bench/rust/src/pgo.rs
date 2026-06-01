//! Ivan-style pose-graph optimization in Rust: SE(3) factor graph (factrs) with
//! odometry between-factors + point-to-plane ICP loop closures and a decoupled
//! rotation/translation loop noise model. Mirrors loop_closure_bench/harness/
//! ivan_pgo.cpp.
use std::collections::HashMap;

use factrs::core::{
    BetweenResidual, GaussNewton, GaussianNoise, Graph, PriorResidual, Values, SE3, SO3,
};
use factrs::traits::*;
use factrs::{assign_symbols, fac};
use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};

use crate::icp;

assign_symbols!(X: SE3);

#[derive(Clone)]
pub struct PgoConfig {
    pub key_pose_delta_trans: f64,
    pub key_pose_delta_deg: f64,
    pub loop_search_radius: f64,
    pub loop_time_thresh: f64,
    pub loop_score_thresh: f64,
    pub loop_submap_half_range: i32,
    pub loop_source_submap_half_range: i32,
    pub submap_resolution: f64,
    pub min_loop_detect_duration: f64,
    pub max_icp_iterations: usize,
    pub max_icp_correspondence_dist: f64,
    pub min_icp_inliers: usize,
    pub min_keyframes_for_loop: usize,
    pub max_loop_offset: f64, // reject loops whose ICP correction exceeds this (0 = off)
}

impl Default for PgoConfig {
    fn default() -> Self {
        PgoConfig {
            key_pose_delta_trans: 0.5,
            key_pose_delta_deg: 10.0,
            loop_search_radius: 2.0,
            loop_time_thresh: 25.0,
            loop_score_thresh: 0.3,
            loop_submap_half_range: 10,
            loop_source_submap_half_range: 0,
            submap_resolution: 0.2,
            min_loop_detect_duration: 5.0,
            max_icp_iterations: 50,
            max_icp_correspondence_dist: 1.0,
            min_icp_inliers: 10,
            min_keyframes_for_loop: 10,
            max_loop_offset: 0.0,
        }
    }
}

struct KeyPose {
    local: Isometry3<f64>,
    optimized: Isometry3<f64>,
    ts: f64,
    cloud: Vec<Vector3<f64>>, // body frame, voxel-downsampled
}

pub struct LoopEdge {
    pub target: usize,
    pub source: usize,
    pub ts_target: f64,
    pub ts_source: f64,
    pub offset: Isometry3<f64>,
    pub score: f64,
}

pub struct Pgo {
    cfg: PgoConfig,
    keyposes: Vec<KeyPose>,
    cache: Vec<LoopEdge>,
    pub history: Vec<LoopEdge>,
    last_loop_ts: f64,
    world_correction: Isometry3<f64>,
}

fn voxel_downsample(pts: &[Vector3<f64>], res: f64) -> Vec<Vector3<f64>> {
    if res <= 0.0 {
        return pts.to_vec();
    }
    let mut acc: HashMap<(i64, i64, i64), (Vector3<f64>, u32)> = HashMap::new();
    for p in pts {
        let key = ((p.x / res).floor() as i64, (p.y / res).floor() as i64, (p.z / res).floor() as i64);
        let e = acc.entry(key).or_insert((Vector3::zeros(), 0));
        e.0 += p;
        e.1 += 1;
    }
    acc.values().map(|(s, c)| s / *c as f64).collect()
}

fn to_se3(iso: &Isometry3<f64>) -> SE3 {
    let t = iso.translation.vector;
    let q = iso.rotation.quaternion();
    SE3::from_rot_trans(SO3::from_xyzw(q.i, q.j, q.k, q.w), Vector3::new(t.x, t.y, t.z))
}

fn from_se3(se: &SE3) -> Isometry3<f64> {
    let xyzw = se.rot().xyzw;
    let uq = UnitQuaternion::from_quaternion(Quaternion::new(xyzw[3], xyzw[0], xyzw[1], xyzw[2]));
    let xyz = se.xyz();
    Isometry3::from_parts(Translation3::new(xyz[0], xyz[1], xyz[2]), uq)
}

impl Pgo {
    pub fn new(cfg: PgoConfig) -> Self {
        Pgo {
            cfg,
            keyposes: Vec::new(),
            cache: Vec::new(),
            history: Vec::new(),
            last_loop_ts: -1.0,
            world_correction: Isometry3::identity(),
        }
    }

    fn is_keyframe(&self, local: &Isometry3<f64>) -> bool {
        match self.keyposes.last() {
            None => true,
            Some(last) => {
                let delta = last.local.inverse() * local;
                let dt = delta.translation.vector.norm();
                let dd = delta.rotation.angle().to_degrees();
                dt > self.cfg.key_pose_delta_trans || dd > self.cfg.key_pose_delta_deg
            }
        }
    }

    pub fn process(&mut self, local: Isometry3<f64>, ts: f64, body_cloud: &[Vector3<f64>]) {
        if body_cloud.is_empty() || !self.is_keyframe(&local) {
            return;
        }
        let cloud = voxel_downsample(body_cloud, self.cfg.submap_resolution);
        let optimized = self.world_correction * local;
        self.keyposes.push(KeyPose { local, optimized, ts, cloud });
        self.search_for_loops();
        self.smooth_and_update();
    }

    fn submap(&self, idx: i32, half: i32) -> Vec<Vector3<f64>> {
        let lo = (idx - half).max(0);
        let hi = (idx + half).min(self.keyposes.len() as i32 - 1);
        let mut out = Vec::new();
        for i in lo..=hi {
            let kp = &self.keyposes[i as usize];
            for p in &kp.cloud {
                out.push(kp.optimized * p);
            }
        }
        voxel_downsample(&out, self.cfg.submap_resolution)
    }

    fn search_for_loops(&mut self) {
        let n = self.keyposes.len();
        if n < self.cfg.min_keyframes_for_loop {
            return;
        }
        let cur_ts = self.keyposes[n - 1].ts;
        if self.last_loop_ts >= 0.0 && cur_ts - self.last_loop_ts < self.cfg.min_loop_detect_duration {
            return;
        }
        let cur_t = self.keyposes[n - 1].optimized.translation.vector;

        let entries: Vec<[f64; 3]> = self.keyposes[..n - 1]
            .iter()
            .map(|kp| {
                let t = kp.optimized.translation.vector;
                [t.x, t.y, t.z]
            })
            .collect();
        let tree: KdTree<f64, 3> = (&entries).into();
        let r2 = self.cfg.loop_search_radius * self.cfg.loop_search_radius;
        let mut found = tree.within::<SquaredEuclidean>(&[cur_t.x, cur_t.y, cur_t.z], r2);
        found.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        let mut loop_idx: i32 = -1;
        for nn in &found {
            let i = nn.item as usize;
            if (cur_ts - self.keyposes[i].ts).abs() > self.cfg.loop_time_thresh {
                loop_idx = i as i32;
                break;
            }
        }
        if loop_idx < 0 {
            return;
        }

        let cur_idx = (n - 1) as i32;
        let target = self.submap(loop_idx, self.cfg.loop_submap_half_range);
        let source = self.submap(cur_idx, self.cfg.loop_source_submap_half_range);
        let res = icp::point_to_plane(
            &source,
            &target,
            self.cfg.max_icp_iterations,
            self.cfg.max_icp_correspondence_dist,
            self.cfg.min_icp_inliers,
        );
        if res.fitness > self.cfg.loop_score_thresh {
            return;
        }
        let cur = &self.keyposes[cur_idx as usize];
        let tgt = &self.keyposes[loop_idx as usize];
        let refined = res.transform * cur.optimized;
        let offset = tgt.optimized.inverse() * refined;
        // Reject false alignments where ICP slid metres along a wall: a true
        // revisit needs only a small correction. (PCL point-to-plane slides
        // less, but the Rust GN point-to-plane can run further into the slide.)
        if self.cfg.max_loop_offset > 0.0
            && offset.translation.vector.norm() > self.cfg.max_loop_offset
        {
            return;
        }
        self.cache.push(LoopEdge {
            target: loop_idx as usize,
            source: cur_idx as usize,
            ts_target: tgt.ts,
            ts_source: cur.ts,
            offset,
            score: res.fitness,
        });
        self.last_loop_ts = cur_ts;
    }

    fn smooth_and_update(&mut self) {
        if self.cache.is_empty() {
            return; // odometry-only: optimized already set via world_correction
        }
        self.history.append(&mut self.cache);

        let mut values = Values::new();
        for (i, kp) in self.keyposes.iter().enumerate() {
            values.insert(X(i as u32), to_se3(&kp.optimized));
        }
        let mut graph = Graph::new();
        // Tight prior pinning keyframe 0 at its odom pose.
        let prior_noise = GaussianNoise::from_split_sigma(1e-6, 1e-6);
        graph.add_factor(fac![PriorResidual::new(to_se3(&self.keyposes[0].local)), X(0), prior_noise]);
        // Odometry between-factors (rot sigma 1e-3, trans sigma 1e-2).
        for i in 1..self.keyposes.len() {
            let delta = self.keyposes[i - 1].local.inverse() * self.keyposes[i].local;
            graph.add_factor(
                fac![BetweenResidual::new(to_se3(&delta)), (X((i - 1) as u32), X(i as u32)), (1e-3, 1e-2) as std],
            );
        }
        // Loop factors: decoupled noise, trans var = ICP fitness (>= 0.01), rot 0.05.
        let rot_var: f64 =
            std::env::var("LOOP_ROT_VAR").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);
        let trans_floor: f64 =
            std::env::var("LOOP_TRANS_FLOOR").ok().and_then(|v| v.parse().ok()).unwrap_or(0.01);
        for lp in &self.history {
            let rot_sig = rot_var.sqrt();
            let trans_sig = lp.score.max(trans_floor).sqrt();
            graph.add_factor(fac![
                BetweenResidual::new(to_se3(&lp.offset)),
                (X(lp.target as u32), X(lp.source as u32)),
                (rot_sig, trans_sig) as std
            ]);
        }

        let mut opt: GaussNewton = GaussNewton::new_default(graph);
        if let Ok(result) = opt.optimize(values) {
            for i in 0..self.keyposes.len() {
                if let Some(se) = result.get::<_, SE3>(X(i as u32)) {
                    self.keyposes[i].optimized = from_se3(se);
                }
            }
        }
        let last = self.keyposes.last().unwrap();
        self.world_correction = last.optimized * last.local.inverse();
    }

    pub fn keyframes(&self) -> Vec<(f64, Isometry3<f64>, Isometry3<f64>)> {
        self.keyposes.iter().map(|kp| (kp.ts, kp.local, kp.optimized)).collect()
    }
}
