//! point-to-plane pose-graph optimization in Rust: SE(3) factor graph (factrs) with
//! odometry between-factors + point-to-plane ICP loop closures and a decoupled
//! rotation/translation loop noise model. Mirrors loop_closure_bench/harness/
//! plane_pgo.cpp.
use std::collections::HashMap;

use factrs::core::{
    BetweenResidual, GaussianNoise, GemanMcClure, Graph, LevenMarquardt, PriorResidual, Values, SE3,
    SO3,
};
use factrs::optimizers::{GncGemanMcClure, GncParams, GraduatedNonConvexity, LevenParams};
use factrs::traits::*;
use factrs::{assign_symbols, fac};
use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::{Isometry3, Matrix6, Quaternion, Translation3, UnitQuaternion, Vector3};

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
    pub loop_rot_var: f64,    // loop rotation noise variance (rad^2)
    pub loop_trans_floor: f64, // loop translation noise variance floor (m^2); higher = distrust loop translation
    pub loop_huber_k: f64,    // (unused legacy) Huber threshold
    pub loop_gm_c: f64,       // Geman-McClure scale (whitened units) when loop_robust=1
    pub loop_robust: bool,    // apply Geman-McClure kernel to loop factors
    pub loop_gnc: bool,       // optimize with Graduated Non-Convexity (robust to bad loops)
    pub gnc_percentile: f64,  // GNC inlier chi-squared percentile (0.95 default)
    pub gnc_mu_step: f64,     // GNC mu graduation step (>1; 1.4 default)
    pub lm_fidelity: f64,     // LM min_model_fidelity (<0 forces step accept; default -1e4)
    pub loop_icp_cov: bool,   // derive loop noise from the ICP Hessian (auto, no trans_floor)
    pub reg_sigma: f64,       // per-point registration noise (m) scaling the ICP information
    pub loop_info_max_sigma: f64, // loosest allowed loop sigma (max distrust on sliding dirs)
    pub loop_info_min_sigma: f64, // tightest allowed loop sigma (cap over-trust on stiff dirs)
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
            loop_rot_var: 0.05,
            loop_trans_floor: 0.01,
            loop_huber_k: 5.0,
            loop_gm_c: 3.0,
            loop_robust: false,
            loop_gnc: false,
            gnc_percentile: 0.95,
            gnc_mu_step: 1.4,
            lm_fidelity: -1e4,
            loop_icp_cov: false,
            reg_sigma: 0.1,
            loop_info_max_sigma: 50.0,
            loop_info_min_sigma: 0.05,
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
    pub icp_t: f64, // magnitude of the ICP correction (≈0 = no slide on clean data)
    pub info: Matrix6<f64>, // ICP plane Hessian (rotation-first); loop information up to 1/σ²
}

pub struct Pgo {
    cfg: PgoConfig,
    keyposes: Vec<KeyPose>,
    cache: Vec<LoopEdge>,
    pub history: Vec<LoopEdge>,
    last_loop_ts: f64,
    world_correction: Isometry3<f64>,
}

pub fn voxel_downsample_pub(pts: &[Vector3<f64>], res: f64) -> Vec<Vector3<f64>> {
    voxel_downsample(pts, res)
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
            icp_t: res.transform.translation.vector.norm(),
            info: res.info,
        });
        self.last_loop_ts = cur_ts;
    }

    // Turn an ICP plane-Hessian into an anisotropic loop noise model. The Hessian
    // already encodes observability: well-constrained directions (rotation, wall
    // normals) have large eigenvalues, the in-plane "sliding" translation has
    // ~zero -- so using it as the factor information automatically distrusts
    // sliding without any hand-set loop_trans_floor. We scale by 1/reg_sigma^2
    // (reg_sigma = per-point registration noise, a sensor property) and clamp the
    // eigenvalues to [1/max_sigma^2, 1/min_sigma^2] for conditioning: the floor
    // makes it positive-definite and caps how much a sliding direction is
    // distrusted; the ceiling stops a many-inlier loop from over-trusting past
    // odometry. All three bounds are dataset/scale-independent.
    fn icp_noise(&self, info: &Matrix6<f64>) -> Option<GaussianNoise<6>> {
        let scaled = info / (self.cfg.reg_sigma * self.cfg.reg_sigma);
        let lo = 1.0 / (self.cfg.loop_info_max_sigma * self.cfg.loop_info_max_sigma);
        let hi = 1.0 / (self.cfg.loop_info_min_sigma * self.cfg.loop_info_min_sigma);
        let mut eig = scaled.symmetric_eigen();
        for i in 0..6 {
            eig.eigenvalues[i] = eig.eigenvalues[i].clamp(lo, hi);
        }
        let clamped = eig.eigenvectors
            * Matrix6::from_diagonal(&eig.eigenvalues)
            * eig.eigenvectors.transpose();
        GaussianNoise::<6>::from_matrix_inf(clamped.as_view())
    }

    // Build the pose graph (prior + odometry + loop factors). `robust` toggles a
    // Geman-McClure kernel on the LOOP factors only (odometry/prior stay L2).
    fn build_graph(&self, robust: bool) -> Graph {
        let mut graph = Graph::new();
        let prior_noise = GaussianNoise::from_split_sigma(1e-6, 1e-6);
        graph.add_factor(fac![PriorResidual::new(to_se3(&self.keyposes[0].local)), X(0), prior_noise]);
        for i in 1..self.keyposes.len() {
            let delta = self.keyposes[i - 1].local.inverse() * self.keyposes[i].local;
            graph.add_factor(
                fac![BetweenResidual::new(to_se3(&delta)), (X((i - 1) as u32), X(i as u32)), (1e-3, 1e-2) as std],
            );
        }
        for lp in &self.history {
            // Preferred path: derive the loop noise from the ICP information matrix
            // (anisotropic, automatic distrust of sliding -- no loop_trans_floor).
            if self.cfg.loop_icp_cov {
                if let Some(noise) = self.icp_noise(&lp.info) {
                    graph.add_factor(fac![
                        BetweenResidual::new(to_se3(&lp.offset)),
                        (X(lp.target as u32), X(lp.source as u32)),
                        noise
                    ]);
                    continue;
                }
            }
            let rot_sig = self.cfg.loop_rot_var.sqrt();
            let trans_sig = lp.score.max(self.cfg.loop_trans_floor).sqrt();
            if robust {
                graph.add_factor(fac![
                    BetweenResidual::new(to_se3(&lp.offset)),
                    (X(lp.target as u32), X(lp.source as u32)),
                    (rot_sig, trans_sig) as std,
                    GemanMcClure::new(self.cfg.loop_gm_c)
                ]);
            } else {
                graph.add_factor(fac![
                    BetweenResidual::new(to_se3(&lp.offset)),
                    (X(lp.target as u32), X(lp.source as u32)),
                    (rot_sig, trans_sig) as std
                ]);
            }
        }
        graph
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

        // Loop translation on the open outdoor scene tends to slide (point-to-
        // plane along the ground plane); the loop-noise knobs (loop_rot_var,
        // loop_trans_floor) let an open-scene config distrust loop translation
        // while trusting rotation, whereas the structured-indoor default trusts
        // both. `robust` (Geman-McClure) is available but off by default.
        // Levenberg-Marquardt with permissive step acceptance: factrs's default
        // min_model_fidelity (1e-3) rejects steps on the stiff/ill-conditioned
        // graphs that tight loop noise produces, so it under-converges (worse than
        // Gauss-Newton). Forcing acceptance (-1e4) + enough iterations makes LM
        // converge well on both the easy indoor graphs and the stiff outdoor ones.
        let mut params = LevenParams::default();
        params.base.max_iterations = 200;
        params.min_model_fidelity = self.cfg.lm_fidelity;
        let log = std::env::var("PGO_LOG").is_ok();
        let g_err = self.build_graph(self.cfg.loop_robust);
        let err_before = if log {
            let mut v = Values::new();
            for (i, kp) in self.keyposes.iter().enumerate() { v.insert(X(i as u32), to_se3(&kp.optimized)); }
            g_err.error(&v)
        } else { 0.0 };
        // Graduated Non-Convexity: the odometry chain (consecutive X(i),X(i+1)
        // factors) is auto-detected as a hard inlier, while loop factors get a
        // Geman-McClure kernel whose mu is graduated from convex to non-convex.
        // This is what makes the batch solve robust at km-scale: a slid/false
        // loop (ICP sliding metres along KITTI's ground plane) is rejected
        // because it disagrees with the trusted odometry, instead of being
        // faithfully applied by the plain L2 solve and corrupting the whole
        // trajectory. Falls back to plain LM when loop_gnc=0.
        let result = if self.cfg.loop_gnc {
            let mut gnc = GncParams::<LevenMarquardt>::default();
            gnc.base.max_iterations = 30; // outer mu-graduation steps
            gnc.mu_step_size = self.cfg.gnc_mu_step;
            gnc.percentile = self.cfg.gnc_percentile;
            // Inner LM uses standard (not step-forcing) acceptance: GNC needs the
            // inner solve to honestly converge each mu so outlier weights are
            // meaningful; the -1e4 fidelity hack is only for the plain-LM path.
            gnc.inner.base.max_iterations = 50;
            let mut opt = GraduatedNonConvexity::<GncGemanMcClure, LevenMarquardt>::new(
                gnc,
                self.build_graph(false),
            );
            opt.optimize(values)
        } else {
            let mut opt = LevenMarquardt::new(params, self.build_graph(self.cfg.loop_robust));
            opt.optimize(values)
        };

        let ok = result.is_ok();
        let solved = match result {
            Ok(v) => Some(v),
            Err(factrs::optimizers::OptError::MaxIterations(v)) => Some(v), // use best-so-far
            Err(_) => None,
        };
        if let Some(result) = solved {
            if log {
                let err_after = g_err.error(&result);
                eprintln!(
                    "[pgo] n_kf={} n_loop={} ok={} err {:.2}->{:.2}",
                    self.keyposes.len(), self.history.len(), ok, err_before, err_after
                );
            }
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
