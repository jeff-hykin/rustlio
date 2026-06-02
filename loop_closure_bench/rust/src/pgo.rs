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
use crate::scan_context::{self, Descriptor, ScanContextConfig};

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
    pub loop_trans_scale: f64, // if >0: loop trans sigma = scale * loop arc length (auto, dynamic)
    pub loop_huber_scale: f64, // ICP point-to-plane Huber delta = scale * submap_resolution (0=off)
    pub min_inlier_ratio: f64, // reject a loop whose ICP overlap is below this (0=off)
    pub loop_fit_max: f64,     // reject loop if fitness/submap_resolution^2 > this (scale-aware; 0=off)
    pub loop_candidates: usize, // ICP-validate the top-N loop candidates, keep best fitness (1=off)
    // Scan Context loop detection (place recognition) instead of spatial-NN. Finds
    // the true revisit under drift, where spatial-NN matches the wrong place.
    pub use_scan_context: bool,
    pub sc_max_range: f64,    // descriptor max range (m); tune to sensor range
    pub sc_dist_thresh: f64,  // SC distance acceptance (0..1)
    pub sc_plus: bool,        // use SC++ cartesian descriptor too
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
            loop_gnc: true,
            gnc_percentile: 0.95,
            gnc_mu_step: 1.4,
            lm_fidelity: -1e4,
            loop_icp_cov: false,
            reg_sigma: 0.1,
            // Max loop translation sigma (loosest / most-distrusted). Also caps the
            // arc-scaled distrust so very long loops -- which carry large drift to
            // correct -- don't get so loose they under-correct (without this,
            // seq02's km loops scale to ~40 m and under-correct; 16 m matches the
            // hand-tuned trans_floor). Generous safety bound, result is insensitive.
            loop_info_max_sigma: 16.0,
            loop_info_min_sigma: 0.05,
            // Opt-in (0 = off, use fixed loop_trans_floor). When >0 (e.g. 0.02),
            // loop translation sigma = clamp(scale*arc, min, max) -- auto/dynamic for
            // open & indoor trajectories, but off by default because it over-distrusts
            // closed-loop (return-home) trajectories. See CONCLUSIONS.md Finding 9.
            loop_trans_scale: 0.0,
            // Default-on robustness (both scale-aware, so they transfer across
            // datasets): Huber down-weights outlier correspondences (delta =
            // 1.0*voxel); loop_fit_max rejects whole false loops whose normalized
            // residual (fitness/voxel^2) exceeds 2.0 -- the discriminator that
            // catches repetitive-structure false matches the overlap gate misses.
            // Validated: improves KITTI, fixes the fastlio-scene corruption,
            // neutral on go2/indoor. Both cheap (real-time-safe).
            loop_huber_scale: 1.0,
            min_inlier_ratio: 0.0,
            loop_fit_max: 2.0,
            loop_candidates: 1,
            use_scan_context: false,
            sc_max_range: 80.0,
            sc_dist_thresh: 0.4,
            sc_plus: true,
        }
    }
}

struct KeyPose {
    local: Isometry3<f64>,
    optimized: Isometry3<f64>,
    ts: f64,
    cloud: Vec<Vector3<f64>>, // body frame, voxel-downsampled
    sc: Option<Descriptor>,   // Scan Context descriptor (body frame); None if disabled
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
    pub arc: f64,   // odometry arc length between the looped keyframes (lever arm, m)
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

    fn sc_config(&self) -> ScanContextConfig {
        let mut c = ScanContextConfig::default();
        c.max_range = self.cfg.sc_max_range;
        c.dist_thresh = self.cfg.sc_dist_thresh;
        c
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
        let sc = if self.cfg.use_scan_context {
            Some(scan_context::compute(&cloud, &self.sc_config()))
        } else {
            None
        };
        self.keyposes.push(KeyPose { local, optimized, ts, cloud, sc });
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
        // Gather candidate target keyframes (time-gated). The true revisit isn't
        // always the nearest spatial / best descriptor (esp. under drift on the
        // Go2 sensor), so we collect the top loop_candidates and ICP-validate each,
        // keeping the best-fitness one -- like PCL/C++ does.
        let k = self.cfg.loop_candidates.max(1);
        let candidates: Vec<usize> = if self.cfg.use_scan_context {
            let cur_sc = self.keyposes[n - 1].sc.as_ref().unwrap();
            let past: Vec<(usize, &Descriptor)> = self.keyposes[..n - 1]
                .iter()
                .enumerate()
                .filter(|(_, kp)| (cur_ts - kp.ts).abs() > self.cfg.loop_time_thresh)
                .filter_map(|(i, kp)| kp.sc.as_ref().map(|d| (i, d)))
                .collect();
            scan_context::top_matches(cur_sc, &past, &self.sc_config(), k)
                .into_iter()
                .map(|(j, _, _)| past[j].0)
                .collect()
        } else {
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
            found
                .iter()
                .map(|nn| nn.item as usize)
                .filter(|&i| (cur_ts - self.keyposes[i].ts).abs() > self.cfg.loop_time_thresh)
                .take(k)
                .collect()
        };

        // ICP-validate each candidate; keep the lowest-fitness one that passes all
        // gates. This is what lets the true (best-aligning) revisit win over a
        // nearer-but-wrong candidate.
        let cur_idx = (n - 1) as i32;
        let mut best: Option<LoopEdge> = None;
        for &loop_idx in &candidates {
            if let Some(edge) = self.eval_candidate(loop_idx as i32, cur_idx) {
                if best.as_ref().map_or(true, |b| edge.score < b.score) {
                    best = Some(edge);
                }
            }
        }
        if let Some(edge) = best {
            self.cache.push(edge);
            self.last_loop_ts = cur_ts;
        }
    }

    // Build submaps for one candidate, run ICP, apply all loop gates. Returns the
    // LoopEdge if it passes, else None.
    fn eval_candidate(&self, loop_idx: i32, cur_idx: i32) -> Option<LoopEdge> {
        let target = self.submap(loop_idx, self.cfg.loop_submap_half_range);
        let source = self.submap(cur_idx, self.cfg.loop_source_submap_half_range);
        let huber = self.cfg.loop_huber_scale * self.cfg.submap_resolution;
        let res = icp::point_to_plane(
            &source,
            &target,
            self.cfg.max_icp_iterations,
            self.cfg.max_icp_correspondence_dist,
            self.cfg.min_icp_inliers,
            huber,
        );
        if res.fitness > self.cfg.loop_score_thresh {
            return None;
        }
        if res.inlier_ratio < self.cfg.min_inlier_ratio {
            return None;
        }
        if self.cfg.loop_fit_max > 0.0
            && res.fitness / (self.cfg.submap_resolution * self.cfg.submap_resolution)
                > self.cfg.loop_fit_max
        {
            return None;
        }
        let cur = &self.keyposes[cur_idx as usize];
        let tgt = &self.keyposes[loop_idx as usize];
        let refined = res.transform * cur.optimized;
        let offset = tgt.optimized.inverse() * refined;
        if self.cfg.max_loop_offset > 0.0
            && offset.translation.vector.norm() > self.cfg.max_loop_offset
        {
            return None;
        }
        let mut arc = 0.0;
        for i in (loop_idx as usize)..(cur_idx as usize) {
            arc += (self.keyposes[i + 1].local.translation.vector
                - self.keyposes[i].local.translation.vector)
                .norm();
        }
        Some(LoopEdge {
            target: loop_idx as usize,
            source: cur_idx as usize,
            ts_target: tgt.ts,
            ts_source: cur.ts,
            offset,
            score: res.fitness,
            icp_t: res.transform.translation.vector.norm(),
            info: res.info,
            arc,
        })
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
            // OPT-IN (loop_trans_scale>0): auto, dynamic translation distrust scaled
            // by the loop's arc length -- trans sigma = clamp(scale*arc, min, max).
            // Replaces the hand-set loop_trans_floor for OPEN/exploratory trajectories
            // (automotive, robots that keep exploring) and indoor: short loops trust
            // translation, long open-traverse loops distrust it, dynamically per-loop
            // with no flags. NOT a universal default -- a trajectory that CLOSES back
            // on itself (returns home) at the same arc wants the opposite (trust), and
            // arc length can't distinguish open from closed (downstream length can't
            // either; it's a global-structure property). So it stays opt-in and the
            // safe default remains the fixed loop_trans_floor. See CONCLUSIONS.md F9.
            let trans_sig = if self.cfg.loop_trans_scale > 0.0 {
                (self.cfg.loop_trans_scale * lp.arc)
                    .clamp(self.cfg.loop_info_min_sigma, self.cfg.loop_info_max_sigma)
            } else {
                lp.score.max(self.cfg.loop_trans_floor).sqrt()
            };
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
