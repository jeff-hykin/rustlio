//! Scan Context place-recognition loop detector (Kim & Kim, IROS 2018).
//!
//! Why: the default detector finds loop candidates by spatial nearest-neighbour
//! in the (drifted) trajectory frame. Under real drift that picks the WRONG place
//! (the revisit has moved metres away while some other leg fell nearer) -- see
//! CONCLUSIONS Finding 10, where the Go2 trajectory's end matched mid-route
//! instead of the start. Scan Context is APPEARANCE-based: it matches a revisit by
//! the shape of its surroundings regardless of where odometry thinks it is.
//!
//! Descriptor: bin each body-frame cloud into `rings` radial x `sectors` azimuth
//! cells, each cell = max point height (z). The row-wise mean is a rotation-
//! invariant "ring key" for fast candidate retrieval; the full matrix is compared
//! with a column-shift-invariant distance whose argmin shift is the relative yaw
//! (a free initial guess for loop ICP).
//!
//! Storage is COLUMN-major (`sector*rings + ring`) so the shift-distance inner
//! product over rings is contiguous -- the row-major version thrashed cache and
//! was ~20x slower.

use nalgebra::Vector3;

#[derive(Clone)]
pub struct ScanContextConfig {
    pub rings: usize,      // radial bins
    pub sectors: usize,    // azimuthal bins
    pub max_range: f64,    // m; points beyond are ignored
    pub ringkey_k: usize,  // # candidates pulled by ring-key search
    pub dist_thresh: f64,  // accept loop if SC distance below this (0..1)
}

impl Default for ScanContextConfig {
    fn default() -> Self {
        ScanContextConfig { rings: 20, sectors: 60, max_range: 80.0, ringkey_k: 10, dist_thresh: 0.3 }
    }
}

#[derive(Clone)]
pub struct Descriptor {
    rings: usize,
    sectors: usize,
    sc: Vec<f32>,          // column-major: sector*rings + ring; cell = max z
    pub ringkey: Vec<f32>, // per-ring mean (rotation-invariant retrieval key)
    colnorm: Vec<f32>,     // per-sector (column) L2 norm, precomputed
}

/// Build the Scan Context descriptor for one body-frame cloud.
pub fn compute(cloud: &[Vector3<f64>], cfg: &ScanContextConfig) -> Descriptor {
    let (nr, ns) = (cfg.rings, cfg.sectors);
    let mut sc = vec![0.0f32; nr * ns];
    let two_pi = std::f64::consts::PI * 2.0;
    for p in cloud {
        let r = (p.x * p.x + p.y * p.y).sqrt();
        if r > cfg.max_range {
            continue;
        }
        let ring = (((r / cfg.max_range) * nr as f64).floor() as usize).min(nr - 1);
        let theta = p.y.atan2(p.x) + std::f64::consts::PI; // [0, 2pi)
        let sec = (((theta / two_pi) * ns as f64).floor() as usize).min(ns - 1);
        let idx = sec * nr + ring; // column-major
        if (p.z as f32) > sc[idx] {
            sc[idx] = p.z as f32;
        }
    }
    let ringkey: Vec<f32> = (0..nr)
        .map(|r| (0..ns).map(|s| sc[s * nr + r]).sum::<f32>() / ns as f32)
        .collect();
    let colnorm: Vec<f32> = (0..ns)
        .map(|s| sc[s * nr..s * nr + nr].iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect();
    Descriptor { rings: nr, sectors: ns, sc, ringkey, colnorm }
}

/// L1 distance between ring keys (cheap, rotation-invariant pre-filter).
pub fn ringkey_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// Column-shift-invariant Scan Context distance. Returns (min_distance,
/// yaw_radians); yaw is the relative heading of `cur` w.r.t. `cand`.
pub fn distance(cur: &Descriptor, cand: &Descriptor) -> (f32, f64) {
    let (nr, ns) = (cur.rings, cur.sectors);
    let (a, b) = (&cur.sc, &cand.sc);
    let (an, bn) = (&cur.colnorm, &cand.colnorm);
    let mut best = f32::INFINITY;
    let mut best_shift = 0usize;
    for shift in 0..ns {
        let mut sum = 0.0f32;
        let mut cnt = 0u32;
        for s in 0..ns {
            let sj = if s + shift >= ns { s + shift - ns } else { s + shift };
            if an[s] < 1e-6 || bn[sj] < 1e-6 {
                continue;
            }
            let ca = &a[s * nr..s * nr + nr];
            let cb = &b[sj * nr..sj * nr + nr];
            let mut dot = 0.0f32;
            for r in 0..nr {
                dot += ca[r] * cb[r];
            }
            sum += 1.0 - dot / (an[s] * bn[sj]);
            cnt += 1;
        }
        if cnt > 0 {
            let d = sum / cnt as f32;
            if d < best {
                best = d;
                best_shift = shift;
            }
        }
    }
    let yaw = -(best_shift as f64) * (std::f64::consts::PI * 2.0 / ns as f64);
    (best, yaw)
}

/// Find the best loop candidate for `cur` among `past` (already time/index-gated
/// by the caller). Two-stage: ring-key L1 to shortlist `ringkey_k`, then full SC
/// distance. Returns (index_into_past, distance, yaw) if below `dist_thresh`.
pub fn best_match(
    cur: &Descriptor,
    past: &[(usize, &Descriptor)],
    cfg: &ScanContextConfig,
) -> Option<(usize, f32, f64)> {
    if past.is_empty() {
        return None;
    }
    let mut shortlist: Vec<(usize, f32)> = past
        .iter()
        .enumerate()
        .map(|(i, (_, d))| (i, ringkey_dist(&cur.ringkey, &d.ringkey)))
        .collect();
    shortlist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    shortlist.truncate(cfg.ringkey_k);

    let mut best: Option<(usize, f32, f64)> = None;
    for (i, _) in shortlist {
        let (d, yaw) = distance(cur, past[i].1);
        if best.is_none() || d < best.unwrap().1 {
            best = Some((i, d, yaw));
        }
    }
    match best {
        Some((i, d, yaw)) if d < cfg.dist_thresh as f32 => Some((i, d, yaw)),
        _ => None,
    }
}
