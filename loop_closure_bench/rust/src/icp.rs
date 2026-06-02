//! Point-to-plane ICP (the core of the point-to-plane loop closure). Normals on the target
//! resolve the "slide along a wall" ambiguity that made naive point-to-point ICP
//! accept multi-metre false alignments.
use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, Vector3};

/// Weight of the point-to-point anchor added to point-to-plane (curbs sliding).
fn p2p_weight() -> f64 {
    std::env::var("P2P_WEIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.15)
}

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Estimate a unit normal at each point via PCA over its k nearest neighbours
/// (smallest-eigenvector of the local covariance).
fn estimate_normals(pts: &[Vector3<f64>], tree: &KdTree<f64, 3>, k: usize) -> Vec<Vector3<f64>> {
    pts.iter()
        .map(|p| {
            let q = [p.x, p.y, p.z];
            let nn = tree.nearest_n::<SquaredEuclidean>(&q, k);
            if nn.len() < 3 {
                return Vector3::z();
            }
            let mut mean = Vector3::zeros();
            for nb in &nn {
                mean += pts[nb.item as usize];
            }
            mean /= nn.len() as f64;
            let mut cov = Matrix3::zeros();
            for nb in &nn {
                let d = pts[nb.item as usize] - mean;
                cov += d * d.transpose();
            }
            let eig = cov.symmetric_eigen();
            // eigenvalues unsorted; pick eigenvector of smallest eigenvalue.
            let mut min_i = 0;
            for i in 1..3 {
                if eig.eigenvalues[i] < eig.eigenvalues[min_i] {
                    min_i = i;
                }
            }
            let n = eig.eigenvectors.column(min_i).into_owned();
            let nn_ = Vector3::new(n.x, n.y, n.z);
            if nn_.norm() < 1e-9 {
                Vector3::z()
            } else {
                nn_.normalize()
            }
        })
        .collect()
}

pub struct IcpResult {
    pub transform: Isometry3<f64>, // aligned = transform * source
    pub fitness: f64,              // mean squared inlier distance (m^2), inf if rejected
    // Plane-only Hessian H = Σ JᵀJ at convergence (rotation-first 6D ordering, in
    // the aligned/world frame). This is the loop measurement's information matrix
    // up to a 1/σ² point-noise scale: poorly-observable directions (in-plane
    // "sliding" translation) appear as near-zero eigenvalues, so using H as the
    // loop factor's information automatically distrusts them -- no hand-tuned
    // loop_trans_floor needed. Excludes the point-to-point anchor (a numerical
    // stabilizer, not real information) so sliding is genuinely uninformative.
    pub info: nalgebra::Matrix6<f64>,
    // Fraction of source points that found an inlier correspondence at convergence
    // (Open3D-style overlap). Low ratio = poor overlap = likely a false loop.
    pub inlier_ratio: f64,
}

/// Align `source` to `target` (both already in a common world frame, init =
/// identity). Returns the correcting transform and the mean-squared inlier
/// distance used as the loop-acceptance fitness.
pub fn point_to_plane(
    source: &[Vector3<f64>],
    target: &[Vector3<f64>],
    max_iter: usize,
    max_dist: f64,
    min_inliers: usize,
    huber_in: f64, // Huber delta (m) on the point-to-plane residual; 0 = off
) -> IcpResult {
    let reject = IcpResult {
        transform: Isometry3::identity(),
        fitness: f64::INFINITY,
        info: nalgebra::Matrix6::zeros(),
        inlier_ratio: 0.0,
    };
    if source.len() < min_inliers || target.len() < min_inliers {
        return reject;
    }
    // Dump every loop's submaps numbered (DUMP_ICP=prefix) so the C++ PCL ICP
    // and this ICP can be compared on byte-identical input, per loop.
    if let Ok(prefix) = std::env::var("DUMP_ICP") {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let i = N.fetch_add(1, Ordering::SeqCst);
        let dump = |path: String, pts: &[Vector3<f64>]| {
            let s: String = pts.iter().map(|p| format!("{} {} {}\n", p.x, p.y, p.z)).collect();
            std::fs::write(path, s).unwrap();
        };
        dump(format!("{prefix}_{i}_src.xyz"), source);
        dump(format!("{prefix}_{i}_tgt.xyz"), target);
    }

    let entries: Vec<[f64; 3]> = target.iter().map(|p| [p.x, p.y, p.z]).collect();
    let tree: KdTree<f64, 3> = (&entries).into();
    let normals = estimate_normals(target, &tree, 12);

    // PCL-style correspondence rejection (env-tunable for A/B):
    //   ICP_NORMAL_COS=c  reject a pair whose source/target normals disagree
    //                     (|n_s·n_t| < c) -- kills wrong-surface matches that
    //                     pass the distance gate and corrupt loops.
    //   ICP_HUBER=d       Huber-weight the point-to-plane residual at delta d (m)
    //                     so a few outlier correspondences can't dominate.
    let reject_cos = std::env::var("ICP_NORMAL_COS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    // param-driven Huber delta; ICP_HUBER env overrides for quick experiments
    let huber = std::env::var("ICP_HUBER").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(huber_in);
    let src_normals: Vec<Vector3<f64>> = if reject_cos > 0.0 {
        let se: Vec<[f64; 3]> = source.iter().map(|p| [p.x, p.y, p.z]).collect();
        let stree: KdTree<f64, 3> = (&se).into();
        estimate_normals(source, &stree, 12)
    } else {
        Vec::new()
    };

    let max_d2 = max_dist * max_dist;
    let w_p2p = p2p_weight();
    let mut rot = Matrix3::identity();
    let mut trans = Vector3::zeros();
    let mut last_fitness = f64::INFINITY;
    let mut last_ratio = 0.0;
    let mut last_info = nalgebra::Matrix6::<f64>::zeros();
    let log = std::env::var("ICP_LOG").is_ok();

    for _iter in 0..max_iter {
        let mut h = nalgebra::Matrix6::<f64>::zeros();
        let mut h_plane = nalgebra::Matrix6::<f64>::zeros(); // plane-only info (no anchor)
        let mut g = nalgebra::Vector6::<f64>::zeros();
        let mut sq_sum = 0.0;
        let mut p2pl_sum = 0.0;
        let mut inliers = 0usize;

        for (si, p) in source.iter().enumerate() {
            let q = rot * p + trans;
            let nn = tree.nearest_one::<SquaredEuclidean>(&[q.x, q.y, q.z]);
            if nn.distance > max_d2 {
                continue;
            }
            let tgt = target[nn.item as usize];
            let n = normals[nn.item as usize];
            // Normal-compatibility rejection: a true surface match has aligned
            // normals; a wrong-surface match (different wall/stair tread) does not.
            if reject_cos > 0.0 {
                let ns = rot * src_normals[si]; // source normal in current frame
                if ns.dot(&n).abs() < reject_cos {
                    continue;
                }
            }
            let a = rot * p; // rotated point R*p (= q - trans)
            let r = n.dot(&(q - tgt)); // point-to-plane residual
            // Huber weight: full weight within delta, downweight outliers ~1/|r|.
            let w = if huber > 0.0 && r.abs() > huber { huber / r.abs() } else { 1.0 };
            // J = [ (a x n)^T , n^T ]  (rotation first, then translation)
            let jrot = a.cross(&n);
            let mut j = nalgebra::Vector6::<f64>::zeros();
            j.fixed_rows_mut::<3>(0).copy_from(&jrot);
            j.fixed_rows_mut::<3>(3).copy_from(&n);
            h += w * j * j.transpose();
            h_plane += w * j * j.transpose();
            g += w * j * r;
            // Small point-to-point anchor. Plane-only ICP can slide freely in a
            // wall plane (zero plane cost); the point-to-point term's minimum on
            // aligned data IS identity, so a light weight pins the slide without
            // hurting plane convergence. J3 = [-skew(a) | I], residual q - tgt.
            let e = q - tgt;
            let mut j3 = nalgebra::Matrix3x6::<f64>::zeros();
            j3.fixed_columns_mut::<3>(0).copy_from(&(-skew(&a)));
            j3.fixed_columns_mut::<3>(3).copy_from(&Matrix3::identity());
            h += w * w_p2p * j3.transpose() * j3;
            g += w * w_p2p * j3.transpose() * e;
            sq_sum += nn.distance; // squared point-to-point distance
            p2pl_sum += r * r; // squared point-to-plane residual
            inliers += 1;
        }

        if inliers < min_inliers {
            return reject;
        }
        last_info = h_plane;
        let fitness = sq_sum / inliers as f64;
        // Tikhonov damping scaled to H's magnitude. For a plane-dominant target
        // the in-plane translation is unobservable (a null space of H); without
        // this the solve runs away by metres along that plane. lambda * I keeps
        // the unobserved directions pinned while barely touching observed ones.
        let lambda = 1e-2 * (h.trace() / 6.0).max(1.0);
        let dx = match (h + nalgebra::Matrix6::identity() * lambda).try_inverse() {
            Some(hinv) => -hinv * g,
            None => return reject,
        };
        let mut drot: Vector3<f64> = dx.fixed_rows::<3>(0).into();
        let mut dtrans: Vector3<f64> = dx.fixed_rows::<3>(3).into();
        // Cap per-iteration step to keep ICP from leaping into a wrong basin.
        let rcap = 0.3; // rad
        let tcap = max_dist; // m
        if drot.norm() > rcap {
            drot *= rcap / drot.norm();
        }
        if dtrans.norm() > tcap {
            dtrans *= tcap / dtrans.norm();
        }
        rot = Rotation3::from_scaled_axis(drot).matrix() * rot;
        trans += dtrans;

        if log {
            eprintln!(
                "  [icp] it={_iter} inliers={inliers}/{} p2p={fitness:.5} p2plane={:.5} |t|={:.3} dt={:.3}",
                source.len(),
                p2pl_sum / inliers as f64,
                trans.norm(),
                dtrans.norm()
            );
        }

        last_fitness = fitness;
        last_ratio = inliers as f64 / source.len() as f64;
        // Converge on a small step (transformation epsilon), like PCL. This
        // stops the slow in-plane crawl early instead of running all iterations.
        if dtrans.norm() < 1e-3 && drot.norm() < 1e-3 {
            break;
        }
    }

    let rotation = Rotation3::from_matrix_unchecked(rot);
    IcpResult {
        transform: Isometry3::from_parts(Translation3::from(trans), rotation.into()),
        fitness: last_fitness,
        info: last_info,
        inlier_ratio: last_ratio,
    }
}
