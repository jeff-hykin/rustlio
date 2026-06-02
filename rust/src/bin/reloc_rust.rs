// Rust port of the global FPFH+RANSAC relocalizer (dimos relocalize.py / the C++
// global_reloc_bench). Global FPFH+RANSAC method, no initial guess:
//   multi-scale FPFH + RANSAC feature matching (restarts per scale)
//   -> 180deg yaw-flip variants -> gravity filter
//   -> wall-only fine-fitness rerank (top-K) -> point-to-plane ICP polish
//   -> final point-to-plane ICP on full fine clouds.
//
// Self-contained (only nalgebra + rayon + std) so it's decoupled from the rest
// of the crate. Implements the reloc_bench CLI contract so bench_reloc.py /
// run/reloc_test can drive it exactly like the C++ backends:
//
//   reloc_rust --map M.pcd --scans S.bin --trials T.txt --out R.txt [k=v ...]
//
// MAP.pcd     binary PCD, fields x y z intensity (f32 packed)
// SCANS.bin   per cloud: [i32 n][n*(x,y,z,i) f32]   (local submaps)
// TRIALS.txt  "scan_idx tx ty tz qx qy qz qw"       (guess IGNORED; global)
// RESULTS.txt "converged tx ty tz qx qy qz qw time_ms"  (converged = fitness gate)
//
// cfg: ransac_iters, restarts_fine, restarts_coarse, accept_fitness, base_res.

use nalgebra::{Matrix3, Matrix4, Vector3};
use rayon::prelude::*;
use std::io::{Read, Write};
use std::time::Instant;

type V3 = Vector3<f64>;
type M4 = Matrix4<f64>;

// Density-adaptive: all distances scale with `base_res` (cfg, default 0.1 m =
// indoor/Livox). Set base_res to the prior map's point spacing (e.g. 0.5 for
// KITTI automotive LiDAR) so FINE_VOXEL, RERANK_DIST and the FPFH scale plan
// track the data density. base_res=0.1 reproduces the original indoor params.
const DEFAULT_BASE_RES: f64 = -1.0; // <0 = auto-estimate from map spacing
const GRAVITY_TILT_MAX_DEG: f64 = 10.0;
const TOP_K: usize = 10;
const NB: usize = 11; // bins per FPFH sub-feature (3 * 11 = 33)

// ----------------------------- 3D kd-tree -----------------------------
struct KdTree {
    pts: Vec<V3>,
    nodes: Vec<KdNode>,
    root: i32,
}
struct KdNode {
    idx: usize,
    axis: u8,
    left: i32,
    right: i32,
}
impl KdTree {
    fn build(pts: Vec<V3>) -> Self {
        let n = pts.len();
        let mut t = KdTree { pts, nodes: Vec::with_capacity(n), root: -1 };
        let mut order: Vec<usize> = (0..n).collect();
        t.root = t.build_rec(&mut order, 0);
        t
    }
    fn build_rec(&mut self, idxs: &mut [usize], depth: usize) -> i32 {
        if idxs.is_empty() {
            return -1;
        }
        let axis = (depth % 3) as u8;
        idxs.sort_by(|&a, &b| {
            self.pts[a][axis as usize].partial_cmp(&self.pts[b][axis as usize]).unwrap()
        });
        let mid = idxs.len() / 2;
        let node_idx = self.nodes.len() as i32;
        self.nodes.push(KdNode { idx: idxs[mid], axis, left: -1, right: -1 });
        let (lo, hi) = idxs.split_at_mut(mid);
        let l = self.build_rec(lo, depth + 1);
        let r = self.build_rec(&mut hi[1..], depth + 1);
        self.nodes[node_idx as usize].left = l;
        self.nodes[node_idx as usize].right = r;
        node_idx
    }
    fn nearest(&self, q: &V3) -> (usize, f64) {
        let mut best = (usize::MAX, f64::MAX);
        self.nearest_rec(self.root, q, &mut best);
        best
    }
    fn nearest_rec(&self, node: i32, q: &V3, best: &mut (usize, f64)) {
        if node < 0 {
            return;
        }
        let nd = &self.nodes[node as usize];
        let p = &self.pts[nd.idx];
        let d2 = (p - q).norm_squared();
        if d2 < best.1 {
            *best = (nd.idx, d2);
        }
        let a = nd.axis as usize;
        let diff = q[a] - p[a];
        let (near, far) = if diff < 0.0 { (nd.left, nd.right) } else { (nd.right, nd.left) };
        self.nearest_rec(near, q, best);
        if diff * diff < best.1 {
            self.nearest_rec(far, q, best);
        }
    }
    // nearest neighbour distance excluding a given index (for spacing estimation)
    fn nearest_skip(&self, q: &V3, skip: usize) -> f64 {
        let mut best = f64::MAX;
        self.nearest_skip_rec(self.root, q, skip, &mut best);
        best
    }
    fn nearest_skip_rec(&self, node: i32, q: &V3, skip: usize, best: &mut f64) {
        if node < 0 {
            return;
        }
        let nd = &self.nodes[node as usize];
        let p = &self.pts[nd.idx];
        if nd.idx != skip {
            let d2 = (p - q).norm_squared();
            if d2 < *best {
                *best = d2;
            }
        }
        let a = nd.axis as usize;
        let diff = q[a] - p[a];
        let (near, far) = if diff < 0.0 { (nd.left, nd.right) } else { (nd.right, nd.left) };
        self.nearest_skip_rec(near, q, skip, best);
        if diff * diff < *best {
            self.nearest_skip_rec(far, q, skip, best);
        }
    }
    fn within(&self, q: &V3, r: f64, out: &mut Vec<usize>) {
        out.clear();
        self.within_rec(self.root, q, r * r, out);
    }
    fn within_rec(&self, node: i32, q: &V3, r2: f64, out: &mut Vec<usize>) {
        if node < 0 {
            return;
        }
        let nd = &self.nodes[node as usize];
        let p = &self.pts[nd.idx];
        if (p - q).norm_squared() <= r2 {
            out.push(nd.idx);
        }
        let a = nd.axis as usize;
        let diff = q[a] - p[a];
        let (near, far) = if diff < 0.0 { (nd.left, nd.right) } else { (nd.right, nd.left) };
        self.within_rec(near, q, r2, out);
        if diff * diff <= r2 {
            self.within_rec(far, q, r2, out);
        }
    }
}

// ----------------------------- geometry -----------------------------
fn voxel_downsample(pts: &[V3], vs: f64) -> Vec<V3> {
    use std::collections::HashMap;
    if vs <= 0.0 {
        return pts.to_vec();
    }
    let inv = 1.0 / vs;
    let mut cells: HashMap<(i64, i64, i64), (V3, u32)> = HashMap::new();
    for p in pts {
        let k = ((p.x * inv).floor() as i64, (p.y * inv).floor() as i64, (p.z * inv).floor() as i64);
        let e = cells.entry(k).or_insert((V3::zeros(), 0));
        e.0 += p;
        e.1 += 1;
    }
    cells.values().map(|(s, c)| s / (*c as f64)).collect()
}

fn estimate_normals(pts: &[V3], tree: &KdTree, radius: f64) -> Vec<V3> {
    pts.par_iter()
        .map(|p| {
            let mut nbr = Vec::new();
            tree.within(p, radius, &mut nbr);
            if nbr.len() < 3 {
                return V3::new(0.0, 0.0, 1.0);
            }
            let mut mean = V3::zeros();
            for &i in &nbr {
                mean += tree.pts[i];
            }
            mean /= nbr.len() as f64;
            let mut cov = Matrix3::zeros();
            for &i in &nbr {
                let d = tree.pts[i] - mean;
                cov += d * d.transpose();
            }
            let eig = cov.symmetric_eigen();
            // smallest eigenvalue's eigenvector = surface normal
            let mut min_i = 0;
            for j in 1..3 {
                if eig.eigenvalues[j] < eig.eigenvalues[min_i] {
                    min_i = j;
                }
            }
            eig.eigenvectors.column(min_i).normalize().into()
        })
        .collect()
}

fn pair_feature(ps: &V3, ns: &V3, pt: &V3, nt: &V3) -> Option<(f64, f64, f64)> {
    let d = pt - ps;
    let dist = d.norm();
    if dist < 1e-9 {
        return None;
    }
    // pick source as the point whose normal aligns better with the connecting line
    let (ps, ns, _pt, nt) = if ns.dot(&d) <= nt.dot(&(-d)) {
        (ps, ns, pt, nt)
    } else {
        (pt, nt, ps, ns)
    };
    let dd = (_pt - ps) / dist;
    let u = *ns;
    let v = u.cross(&dd);
    let vn = v.norm();
    if vn < 1e-9 {
        return None;
    }
    let v = v / vn;
    let w = u.cross(&v);
    let f1 = v.dot(nt); // alpha [-1,1]
    let f2 = u.dot(&dd); // phi [-1,1]
    let f3 = w.dot(nt).atan2(u.dot(nt)); // theta [-pi,pi]
    Some((f1, f2, f3))
}

fn compute_fpfh(pts: &[V3], normals: &[V3], tree: &KdTree, radius: f64) -> Vec<[f32; 33]> {
    let pi = std::f64::consts::PI;
    // SPFH per point
    let spfh: Vec<[f64; 33]> = (0..pts.len())
        .into_par_iter()
        .map(|i| {
            let mut nbr = Vec::new();
            tree.within(&pts[i], radius, &mut nbr);
            let mut h = [0.0f64; 33];
            let mut cnt = 0.0;
            for &j in &nbr {
                if j == i {
                    continue;
                }
                if let Some((f1, f2, f3)) = pair_feature(&pts[i], &normals[i], &pts[j], &normals[j]) {
                    let b1 = (((f1 + 1.0) / 2.0 * NB as f64) as usize).min(NB - 1);
                    let b2 = (((f2 + 1.0) / 2.0 * NB as f64) as usize).min(NB - 1);
                    let b3 = (((f3 + pi) / (2.0 * pi) * NB as f64) as usize).min(NB - 1);
                    h[b1] += 1.0;
                    h[NB + b2] += 1.0;
                    h[2 * NB + b3] += 1.0;
                    cnt += 1.0;
                }
            }
            if cnt > 0.0 {
                // normalize each 11-bin sub-histogram to sum 100 (Open3D convention)
                for blk in 0..3 {
                    let s: f64 = h[blk * NB..blk * NB + NB].iter().sum();
                    if s > 0.0 {
                        for b in 0..NB {
                            h[blk * NB + b] *= 100.0 / s;
                        }
                    }
                }
            }
            h
        })
        .collect();

    // FPFH = SPFH_i + (1/sum_w) * sum_j w_j SPFH_j over radius neighbours
    (0..pts.len())
        .into_par_iter()
        .map(|i| {
            let mut nbr = Vec::new();
            tree.within(&pts[i], radius, &mut nbr);
            let mut f = spfh[i];
            let mut acc = [0.0f64; 33];
            let mut sw = 0.0;
            for &j in &nbr {
                if j == i {
                    continue;
                }
                let dist = (pts[i] - pts[j]).norm();
                if dist < 1e-9 {
                    continue;
                }
                let w = 1.0 / dist;
                sw += w;
                for d in 0..33 {
                    acc[d] += w * spfh[j][d];
                }
            }
            if sw > 0.0 {
                for d in 0..33 {
                    f[d] += acc[d] / sw;
                }
            }
            let mut out = [0.0f32; 33];
            for d in 0..33 {
                out[d] = f[d] as f32;
            }
            out
        })
        .collect()
}

// brute-force nearest target feature for each source feature (33-D); rayon over source
fn feature_correspondences(src: &[[f32; 33]], tgt: &[[f32; 33]]) -> Vec<usize> {
    src.par_iter()
        .map(|s| {
            let mut best = (usize::MAX, f32::MAX);
            for (j, t) in tgt.iter().enumerate() {
                let mut d = 0.0f32;
                for k in 0..33 {
                    let e = s[k] - t[k];
                    d += e * e;
                    if d >= best.1 {
                        break;
                    }
                }
                if d < best.1 {
                    best = (j, d);
                }
            }
            best.0
        })
        .collect()
}

// rigid transform aligning src points to dst points (Kabsch / Umeyama, no scale)
fn kabsch(src: &[V3], dst: &[V3]) -> M4 {
    let n = src.len() as f64;
    let mut cs = V3::zeros();
    let mut cd = V3::zeros();
    for i in 0..src.len() {
        cs += src[i];
        cd += dst[i];
    }
    cs /= n;
    cd /= n;
    let mut h = Matrix3::zeros();
    for i in 0..src.len() {
        h += (src[i] - cs) * (dst[i] - cd).transpose();
    }
    let svd = h.svd(true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let mut d = Matrix3::identity();
    if (vt.transpose() * u.transpose()).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let r = vt.transpose() * d * u.transpose();
    let t = cd - r * cs;
    let mut m = M4::identity();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    m.fixed_view_mut::<3, 1>(0, 3).copy_from(&t);
    m
}

fn transform(m: &M4, p: &V3) -> V3 {
    let r = m.fixed_view::<3, 3>(0, 0);
    let t = m.fixed_view::<3, 1>(0, 3);
    r * p + t
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[allow(clippy::too_many_arguments)]
fn ransac(
    src: &[V3],
    tgt: &[V3],
    _tgt_tree: &KdTree,
    corr: &[usize],
    max_corr: f64,
    iters: usize,
    seed: u64,
) -> M4 {
    // valid correspondence indices
    let valid: Vec<usize> = (0..src.len()).filter(|&i| corr[i] != usize::MAX).collect();
    if valid.len() < 3 {
        return M4::identity();
    }
    let mut rng = Rng(seed | 1);
    let mut best_t = M4::identity();
    let mut best_inl = 0usize;
    let edge_lo = 0.9;
    let max_corr2 = max_corr * max_corr;
    let mut s = [V3::zeros(); 3];
    let mut d = [V3::zeros(); 3];
    for _ in 0..iters {
        // sample 3 distinct correspondences
        let (a, b, c) = (
            valid[rng.range(valid.len())],
            valid[rng.range(valid.len())],
            valid[rng.range(valid.len())],
        );
        if a == b || b == c || a == c {
            continue;
        }
        let si = [a, b, c];
        for k in 0..3 {
            s[k] = src[si[k]];
            d[k] = tgt[corr[si[k]]];
        }
        // edge-length prerejection (similar polygon)
        let mut ok = true;
        for (k, l) in [(0, 1), (1, 2), (0, 2)] {
            let ls = (s[k] - s[l]).norm();
            let ld = (d[k] - d[l]).norm();
            if ls < 1e-6 || ld < 1e-6 || ls.min(ld) / ls.max(ld) < edge_lo {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        let t = kabsch(&s, &d);
        // correspondence-based inlier count: a correspondence is an inlier if
        // T*src_i lands near its matched target tgt[corr_i] (O(1), no tree) —
        // standard for feature RANSAC and ~1000x faster than a geometric search.
        let mut inl = 0usize;
        for &i in &valid {
            let p = transform(&t, &src[i]);
            if (p - tgt[corr[i]]).norm_squared() <= max_corr2 {
                inl += 1;
            }
        }
        if inl > best_inl {
            best_inl = inl;
            best_t = t;
        }
    }
    best_t
}

fn gravity_tilt_deg(t: &M4) -> f64 {
    let z = t.fixed_view::<3, 3>(0, 0) * V3::new(0.0, 0.0, 1.0);
    z.z.clamp(-1.0, 1.0).acos().to_degrees()
}

fn inlier_fitness(src: &[V3], tgt_tree: &KdTree, t: &M4, dist: f64) -> f64 {
    if src.is_empty() {
        return 0.0;
    }
    let d2 = dist * dist;
    let inl = src
        .par_iter()
        .filter(|p| {
            let q = transform(t, p);
            tgt_tree.nearest(&q).1 <= d2
        })
        .count();
    inl as f64 / src.len() as f64
}

// point-to-plane ICP: align src points to tgt (with normals) starting at init
fn point_to_plane_icp(
    src: &[V3],
    tgt: &[V3],
    tgt_n: &[V3],
    tgt_tree: &KdTree,
    init: &M4,
    max_corr: f64,
    iters: usize,
) -> M4 {
    let mut t = *init;
    let max_corr2 = max_corr * max_corr;
    for _ in 0..iters {
        // accumulate 6x6 normal equations (point-to-plane, se3 left-perturbation)
        let mut ata = nalgebra::Matrix6::<f64>::zeros();
        let mut atb = nalgebra::Vector6::<f64>::zeros();
        let mut used = 0;
        for sp in src {
            let p = transform(&t, sp);
            let (j, d2) = tgt_tree.nearest(&p);
            if d2 > max_corr2 {
                continue;
            }
            let n = tgt_n[j];
            let q = tgt[j];
            let r = (p - q).dot(&n);
            // jacobian: d(r)/d[dtheta(3), dt(3)] = [ (p x n)^T , n^T ]
            let pxn = p.cross(&n);
            let mut jrow = nalgebra::Vector6::<f64>::zeros();
            jrow.fixed_rows_mut::<3>(0).copy_from(&pxn);
            jrow.fixed_rows_mut::<3>(3).copy_from(&n);
            ata += jrow * jrow.transpose();
            atb -= jrow * r;
            used += 1;
        }
        if used < 6 {
            break;
        }
        let delta = match ata.lu().solve(&atb) {
            Some(x) => x,
            None => break,
        };
        let dtheta = V3::new(delta[0], delta[1], delta[2]);
        let dt = V3::new(delta[3], delta[4], delta[5]);
        let dr = so3_exp(&dtheta);
        let mut newt = M4::identity();
        let r_old = t.fixed_view::<3, 3>(0, 0).into_owned();
        let t_old = t.fixed_view::<3, 1>(0, 3).into_owned();
        newt.fixed_view_mut::<3, 3>(0, 0).copy_from(&(dr * r_old));
        newt.fixed_view_mut::<3, 1>(0, 3).copy_from(&(dr * t_old + dt));
        t = newt;
        if dtheta.norm() < 1e-5 && dt.norm() < 1e-5 {
            break;
        }
    }
    t
}

fn so3_exp(w: &V3) -> Matrix3<f64> {
    let theta = w.norm();
    if theta < 1e-9 {
        return Matrix3::identity();
    }
    let k = w / theta;
    let kx = Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
    Matrix3::identity() + theta.sin() * kx + (1.0 - theta.cos()) * (kx * kx)
}

// ----------------------------- target cache -----------------------------
struct ScaleData {
    down: Vec<V3>,
    fpfh: Vec<[f32; 33]>,
    tree: KdTree,
}
struct Target {
    scales: Vec<(f64, usize)>,
    fine_voxel: f64,
    rerank: f64,
    sdata: Vec<ScaleData>,
    fine: Vec<V3>,
    fine_n: Vec<V3>,
    fine_tree: KdTree,
    wall: Vec<V3>,
    wall_n: Vec<V3>,
    wall_tree: KdTree,
}

fn wall_subset(pts: &[V3], n: &[V3]) -> (Vec<V3>, Vec<V3>) {
    let mut wp = Vec::new();
    let mut wn = Vec::new();
    for i in 0..pts.len() {
        if n[i].z.abs() < 0.7 {
            wp.push(pts[i]);
            wn.push(n[i]);
        }
    }
    if wp.len() < 100 {
        (pts.to_vec(), n.to_vec())
    } else {
        (wp, wn)
    }
}

fn preprocess(pts: &[V3], vs: f64) -> (Vec<V3>, Vec<V3>, KdTree, Vec<[f32; 33]>) {
    let down = voxel_downsample(pts, vs);
    let tree = KdTree::build(down.clone());
    let normals = estimate_normals(&down, &tree, vs * 2.0);
    let fpfh = compute_fpfh(&down, &normals, &tree, vs * 5.0);
    (down, normals, tree, fpfh)
}

// Estimate the map's point spacing (median nearest-neighbour distance over a
// sample). Used to auto-set base_res so the pipeline adapts to sensor density.
fn estimate_spacing(pts: &[V3]) -> f64 {
    if pts.len() < 10 {
        return 0.1;
    }
    let tree = KdTree::build(pts.to_vec());
    let step = (pts.len() / 3000).max(1);
    let mut ds: Vec<f64> = (0..pts.len())
        .step_by(step)
        .map(|i| tree.nearest_skip(&pts[i], i).sqrt())
        .filter(|d| d.is_finite() && *d > 1e-6)
        .collect();
    if ds.is_empty() {
        return 0.1;
    }
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ds[ds.len() / 2]
}

fn build_target(map: &[V3], scales: Vec<(f64, usize)>, fine_voxel: f64, rerank: f64) -> Target {
    let mut sdata = Vec::new();
    for &(vs, _) in &scales {
        let (down, _n, tree, fpfh) = preprocess(map, vs);
        sdata.push(ScaleData { down, fpfh, tree });
    }
    let fine = voxel_downsample(map, fine_voxel);
    let fine_tree = KdTree::build(fine.clone());
    let fine_n = estimate_normals(&fine, &fine_tree, fine_voxel * 2.0);
    let (wall, wall_n) = wall_subset(&fine, &fine_n);
    let wall_tree = KdTree::build(wall.clone());
    Target { scales, fine_voxel, rerank, sdata, fine, fine_n, fine_tree, wall, wall_n, wall_tree }
}

fn relocalize(local: &[V3], tc: &Target, ransac_iters: usize, seed: u64) -> (M4, f64) {
    let src_fine = voxel_downsample(local, tc.fine_voxel);
    let src_fine_tree = KdTree::build(src_fine.clone());
    let src_fine_n = estimate_normals(&src_fine, &src_fine_tree, tc.fine_voxel * 2.0);
    let (src_wall, src_wall_n) = wall_subset(&src_fine, &src_fine_n);

    let mut candidates: Vec<M4> = Vec::new();
    for (si, &(vs, restarts)) in tc.scales.iter().enumerate() {
        let (down, _n, _tree, fpfh) = preprocess(local, vs);
        let corr = feature_correspondences(&fpfh, &tc.sdata[si].fpfh);
        for r in 0..restarts {
            let t = ransac(
                &down,
                &tc.sdata[si].down,
                &tc.sdata[si].tree,
                &corr,
                vs * 1.5,
                ransac_iters,
                seed.wrapping_add((si as u64) << 32).wrapping_add(r as u64 + 1),
            );
            candidates.push(t);
        }
    }

    // 180-deg yaw flip about submap xy-centroid
    let mut c = V3::zeros();
    for p in &src_fine {
        c += p;
    }
    c /= src_fine.len().max(1) as f64;
    let mut flip = M4::identity();
    flip[(0, 0)] = -1.0;
    flip[(1, 1)] = -1.0;
    flip[(0, 3)] = 2.0 * c.x;
    flip[(1, 3)] = 2.0 * c.y;
    let base = candidates.len();
    for i in 0..base {
        candidates.push(candidates[i] * flip);
    }

    // gravity filter
    let mut pool: Vec<M4> = candidates.iter().cloned().filter(|t| gravity_tilt_deg(t) <= GRAVITY_TILT_MAX_DEG).collect();
    if pool.is_empty() {
        pool = candidates;
    }

    // rerank by wall-only fine fitness, top-K
    let mut scored: Vec<(f64, M4)> =
        pool.iter().map(|t| (inlier_fitness(&src_wall, &tc.wall_tree, t, tc.rerank), *t)).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored.truncate(TOP_K);

    // polish each on walls, pick best
    let mut best_fit = -1.0;
    let mut best_t = M4::identity();
    for (_, t0) in &scored {
        let t = point_to_plane_icp(&src_wall, &tc.wall, &tc.wall_n, &tc.wall_tree, t0, tc.rerank, 70);
        let fit = inlier_fitness(&src_wall, &tc.wall_tree, &t, tc.rerank);
        if fit > best_fit {
            best_fit = fit;
            best_t = t;
        }
    }

    // final ICP on full fine clouds
    let final_t = point_to_plane_icp(&src_fine, &tc.fine, &tc.fine_n, &tc.fine_tree, &best_t, tc.rerank, 50);
    (final_t, best_fit)
}

// ----------------------------- IO -----------------------------
fn read_pcd_binary(path: &str) -> Vec<V3> {
    let mut f = std::fs::File::open(path).expect("open pcd");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    // find end of header ("DATA binary\n")
    let header_end = {
        let needle = b"DATA binary\n";
        let pos = buf.windows(needle.len()).position(|w| w == needle).expect("DATA binary header");
        pos + needle.len()
    };
    let mut npts = 0usize;
    for line in std::str::from_utf8(&buf[..header_end]).unwrap_or("").lines() {
        if let Some(rest) = line.strip_prefix("POINTS ") {
            npts = rest.trim().parse().unwrap_or(0);
        }
    }
    let body = &buf[header_end..];
    let mut out = Vec::with_capacity(npts);
    let mut o = 0;
    for _ in 0..npts {
        if o + 16 > body.len() {
            break;
        }
        let x = f32::from_le_bytes(body[o..o + 4].try_into().unwrap());
        let y = f32::from_le_bytes(body[o + 4..o + 8].try_into().unwrap());
        let z = f32::from_le_bytes(body[o + 8..o + 12].try_into().unwrap());
        out.push(V3::new(x as f64, y as f64, z as f64));
        o += 16;
    }
    out
}

fn read_scans(path: &str) -> Vec<Vec<V3>> {
    let mut f = std::fs::File::open(path).expect("open scans");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let mut scans = Vec::new();
    let mut o = 0;
    while o + 4 <= buf.len() {
        let n = i32::from_le_bytes(buf[o..o + 4].try_into().unwrap()) as usize;
        o += 4;
        let mut cloud = Vec::with_capacity(n);
        for _ in 0..n {
            if o + 16 > buf.len() {
                break;
            }
            let x = f32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
            let y = f32::from_le_bytes(buf[o + 4..o + 8].try_into().unwrap());
            let z = f32::from_le_bytes(buf[o + 8..o + 12].try_into().unwrap());
            cloud.push(V3::new(x as f64, y as f64, z as f64));
            o += 16;
        }
        scans.push(cloud);
    }
    scans
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut map = String::new();
    let mut scans = String::new();
    let mut trials = String::new();
    let mut out = String::new();
    let mut ransac_iters = 100_000usize;
    let mut restarts_fine = 8usize;
    let mut restarts_coarse = 1usize;
    let mut accept_fitness = 0.15f64;
    let mut base_res = DEFAULT_BASE_RES;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        let mut nextv = || {
            i += 1;
            args[i].clone()
        };
        match a.as_str() {
            "--map" => map = nextv(),
            "--scans" => scans = nextv(),
            "--trials" => trials = nextv(),
            "--out" => out = nextv(),
            _ => {
                if let Some((k, v)) = a.split_once('=') {
                    match k {
                        "ransac_iters" => ransac_iters = v.parse().unwrap(),
                        "restarts_fine" => restarts_fine = v.parse().unwrap(),
                        "restarts_coarse" => restarts_coarse = v.parse().unwrap(),
                        "accept_fitness" => accept_fitness = v.parse().unwrap(),
                        "base_res" => base_res = v.parse().unwrap(),
                        _ => {}
                    }
                }
            }
        }
        i += 1;
    }
    if map.is_empty() || scans.is_empty() || trials.is_empty() || out.is_empty() {
        eprintln!("usage: reloc_rust --map M.pcd --scans S.bin --trials T.txt --out R.txt [k=v]");
        std::process::exit(1);
    }

    let map_pts = read_pcd_binary(&map);
    eprintln!("[reloc_rust] building target from {} map points...", map_pts.len());
    let tb = Instant::now();
    // Auto-estimate base_res from the map's point spacing unless overridden.
    if base_res <= 0.0 {
        base_res = estimate_spacing(&map_pts).clamp(0.05, 1.0);
        eprintln!("[reloc_rust] auto base_res = {:.3} (map spacing)", base_res);
    }
    // Density-adaptive scale plan / distances, all proportional to base_res
    // (0.1 -> original indoor 0.2/0.3/0.8 & fine 0.1; 0.5 -> 1.0/1.5/4.0 & fine 0.5).
    let fine_voxel = base_res;
    let rerank = base_res * 1.5;
    let scales = vec![
        (base_res * 2.0, restarts_fine),
        (base_res * 3.0, restarts_fine),
        (base_res * 8.0, restarts_coarse),
    ];
    let tc = build_target(&map_pts, scales, fine_voxel, rerank);
    eprintln!(
        "[reloc_rust] base_res={} target ready (fine={} walls={}) in {:.2}s",
        base_res,
        tc.fine.len(),
        tc.wall.len(),
        tb.elapsed().as_secs_f64()
    );
    let submaps = read_scans(&scans);
    eprintln!("[reloc_rust] loaded {} submaps", submaps.len());

    let trials_txt = std::fs::read_to_string(&trials).unwrap();
    let mut of = std::fs::File::create(&out).unwrap();
    let (mut n, mut acc) = (0usize, 0usize);
    for line in trials_txt.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let idx: usize = f[0].parse().unwrap();
        // guess (f[1..8]) ignored — global method
        let t0 = Instant::now();
        let (t, fit) = relocalize(&submaps[idx], &tc, ransac_iters, 0x9E3779B97F4A7C15 ^ (idx as u64));
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let r = t.fixed_view::<3, 3>(0, 0).into_owned();
        let q = nalgebra::UnitQuaternion::from_matrix(&r);
        let tt = t.fixed_view::<3, 1>(0, 3);
        let ok = fit >= accept_fitness;
        writeln!(
            of,
            "{} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.4}",
            if ok { 1 } else { 0 },
            tt[0], tt[1], tt[2], q.i, q.j, q.k, q.w, ms
        )
        .unwrap();
        n += 1;
        if ok {
            acc += 1;
        }
        eprintln!("  trial {}/idx{} fitness={:.4} {:.0}ms", n, idx, fit, ms);
    }
    eprintln!("[reloc_rust] trials={} accepted={}", n, acc);
}
