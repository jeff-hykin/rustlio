//! Rust port of the C++ pgo_bench harness driving the Rust Ivan-style PGO.
//! Same CLI contract: `--clouds f --poses f --out f [key=val ...]`.
mod icp;
mod io;
mod pgo;

use std::collections::HashMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
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
    eprintln!("[pgo_bench_rs] keyframes={} loops={}", kfs.len(), loops.len());
    io::write_json(&out, &kfs, &loops);
    eprintln!("[pgo_bench_rs] wrote {}", out);
}
