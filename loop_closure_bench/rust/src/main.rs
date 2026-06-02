//! Rust port of the C++ pgo_bench harness driving the Rust point-to-plane PGO.
//! Same CLI contract: `--clouds f --poses f --out f [key=val ...]`.
mod icp;
mod io;
mod pgo;
mod scan_context;

use std::collections::HashMap;

fn load_xyz(path: &str) -> Vec<nalgebra::Vector3<f64>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().map(|s| s.parse().unwrap()).collect();
            nalgebra::Vector3::new(v[0], v[1], v[2])
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Diagnostic: build a submap from frames [idx-half, idx+half] of clouds.bin
    // (transform each body cloud by its TUM pose, merge, voxel-downsample) and
    // dump it + print the count. Mirrors the C++ for an apples-to-apples compare.
    if args.len() >= 7 && args[1] == "--submap-test" {
        let frames = io::load(&args[3], &args[2]); // poses, clouds
        let idx: i64 = args[4].parse().unwrap();
        let half: i64 = args[5].parse().unwrap();
        let res: f64 = args[6].parse().unwrap();
        let lo = (idx - half).max(0) as usize;
        let hi = ((idx + half) as usize).min(frames.len() - 1);
        let mut out = Vec::new();
        for i in lo..=hi {
            for p in &frames[i].cloud {
                out.push(frames[i].pose * p);
            }
        }
        let raw = out.len();
        let ds = pgo::voxel_downsample_pub(&out, res);
        let s: String = ds.iter().map(|p| format!("{} {} {}\n", p.x, p.y, p.z)).collect();
        std::fs::write("/tmp/rust_submap.xyz", s).unwrap();
        println!("rust_submap: frames[{lo}..{hi}] raw={raw} downsampled={}", ds.len());
        return;
    }

    // Diagnostic: run this ICP on two dumped submaps, print the transform.
    if args.len() >= 4 && args[1] == "--icp-test" {
        let src = load_xyz(&args[2]);
        let tgt = load_xyz(&args[3]);
        let r = icp::point_to_plane(&src, &tgt, 50, 1.0, 10, 0.0);
        let t = r.transform.translation.vector;
        println!(
            "rust_icp: |t|={:.4} t=[{:.4},{:.4},{:.4}] fitness={:.5}",
            t.norm(), t.x, t.y, t.z, r.fitness
        );
        return;
    }
    let (mut clouds, mut poses, mut out) = (String::new(), String::new(), String::new());
    let mut kv: HashMap<String, String> = HashMap::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--clouds" => { clouds = args[i + 1].clone(); i += 2; }
            "--poses" => { poses = args[i + 1].clone(); i += 2; }
            "--out" => { out = args[i + 1].clone(); i += 2; }
            a => {
                if let Some(eq) = a.find('=') {
                    kv.insert(a[..eq].to_string(), a[eq + 1..].to_string());
                }
                i += 1;
            }
        }
    }
    if clouds.is_empty() || poses.is_empty() || out.is_empty() {
        eprintln!("usage: pgo_bench_rs --clouds f --poses f --out f [key=val ...]");
        std::process::exit(1);
    }

    let mut cfg = pgo::PgoConfig::default();
    let getd = |k: &str, d: f64| kv.get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
    let geti = |k: &str, d: i32| kv.get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
    cfg.key_pose_delta_deg = getd("key_pose_delta_deg", cfg.key_pose_delta_deg);
    cfg.key_pose_delta_trans = getd("key_pose_delta_trans", cfg.key_pose_delta_trans);
    cfg.loop_search_radius = getd("loop_search_radius", cfg.loop_search_radius);
    // C++ harness spells it loop_time_tresh; accept both.
    cfg.loop_time_thresh = getd("loop_time_thresh", getd("loop_time_tresh", cfg.loop_time_thresh));
    cfg.loop_score_thresh = getd("loop_score_thresh", getd("loop_score_tresh", cfg.loop_score_thresh));
    cfg.loop_submap_half_range = geti("loop_submap_half_range", cfg.loop_submap_half_range);
    cfg.loop_source_submap_half_range =
        geti("loop_source_submap_half_range", cfg.loop_source_submap_half_range);
    cfg.submap_resolution = getd("submap_resolution", cfg.submap_resolution);
    cfg.min_loop_detect_duration = getd("min_loop_detect_duration", cfg.min_loop_detect_duration);
    cfg.max_icp_correspondence_dist = getd("max_icp_correspondence_dist", cfg.max_icp_correspondence_dist);
    cfg.max_loop_offset = getd("max_loop_offset", cfg.max_loop_offset);
    cfg.max_icp_iterations = geti("max_icp_iterations", cfg.max_icp_iterations as i32) as usize;
    cfg.loop_rot_var = getd("loop_rot_var", cfg.loop_rot_var);
    cfg.loop_trans_floor = getd("loop_trans_floor", cfg.loop_trans_floor);
    cfg.loop_huber_k = getd("loop_huber_k", cfg.loop_huber_k);
    cfg.loop_gm_c = getd("loop_gm_c", cfg.loop_gm_c);
    cfg.loop_robust = geti("loop_robust", cfg.loop_robust as i32) != 0;
    cfg.loop_gnc = geti("loop_gnc", cfg.loop_gnc as i32) != 0;
    cfg.gnc_percentile = getd("gnc_percentile", cfg.gnc_percentile);
    cfg.gnc_mu_step = getd("gnc_mu_step", cfg.gnc_mu_step);
    cfg.lm_fidelity = getd("lm_fidelity", cfg.lm_fidelity);
    cfg.loop_icp_cov = geti("loop_icp_cov", cfg.loop_icp_cov as i32) != 0;
    cfg.reg_sigma = getd("reg_sigma", cfg.reg_sigma);
    cfg.loop_info_max_sigma = getd("loop_info_max_sigma", cfg.loop_info_max_sigma);
    cfg.loop_info_min_sigma = getd("loop_info_min_sigma", cfg.loop_info_min_sigma);
    cfg.loop_trans_scale = getd("loop_trans_scale", cfg.loop_trans_scale);
    cfg.loop_huber_scale = getd("loop_huber_scale", cfg.loop_huber_scale);
    cfg.min_inlier_ratio = getd("min_inlier_ratio", cfg.min_inlier_ratio);
    cfg.loop_fit_max = getd("loop_fit_max", cfg.loop_fit_max);
    cfg.loop_candidates = geti("loop_candidates", cfg.loop_candidates as i32) as usize;
    cfg.use_scan_context = geti("use_scan_context", cfg.use_scan_context as i32) != 0;
    cfg.sc_max_range = getd("sc_max_range", cfg.sc_max_range);
    cfg.sc_dist_thresh = getd("sc_dist_thresh", cfg.sc_dist_thresh);
    cfg.sc_plus = geti("sc_plus", cfg.sc_plus as i32) != 0;

    let frames = io::load(&poses, &clouds);
    eprintln!("[pgo_bench_rs] loaded {} frames", frames.len());

    let mut graph = pgo::Pgo::new(cfg);
    for fr in &frames {
        graph.process(fr.pose, fr.ts, &fr.cloud);
    }

    let kfs: Vec<io::OutKeyframe> = graph
        .keyframes()
        .into_iter()
        .map(|(ts, raw, opt)| io::OutKeyframe { ts, raw, opt })
        .collect();
    let loops: Vec<io::OutLoop> = graph
        .history
        .iter()
        .map(|l| io::OutLoop {
            target: l.target,
            source: l.source,
            ts_target: l.ts_target,
            ts_source: l.ts_source,
            score: l.score,
            offset_t: l.offset.translation.vector.norm(),
        })
        .collect();
    if std::env::var("ICP_LOG").is_ok() {
        for l in &graph.history {
            eprintln!("[rust loop] src={} tgt={} icp_t={:.3} arc={:.1}", l.source, l.target, l.icp_t, l.arc);
        }
    }
    eprintln!("[pgo_bench_rs] keyframes={} loops={}", kfs.len(), loops.len());
    io::write_json(&out, &kfs, &loops);
    eprintln!("[pgo_bench_rs] wrote {}", out);
}
