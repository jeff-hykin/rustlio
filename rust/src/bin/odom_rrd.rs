//! Combine several odom `.npy` files (as written by the `fastlio2` binary:
//! columns `[t, x, y, z, vx, vy, vz]`) into a single Rerun `.rrd` recording,
//! logging each run's trajectory as a distinctly-colored 3D line + points so
//! all runs can be compared in one 3D view.
//!
//! Usage: `odom_rrd <out.rrd> <run0.npy> [run1.npy ...]`

use ndarray::Array2;
use ndarray_npy::read_npy;

const PALETTE: [(u8, u8, u8); 10] = [
    (231, 76, 60), (230, 126, 34), (241, 196, 15), (46, 204, 113), (52, 152, 219),
    (155, 89, 182), (26, 188, 156), (233, 30, 99), (0, 188, 212), (255, 87, 34),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: odom_rrd <out.rrd> <run0.npy> [run1.npy ...]");
        std::process::exit(2);
    }
    let out = &args[1];
    let files = &args[2..];

    // No config file here; default to Normal (override via RUST_LOG).
    rustlio::logging::init(rustlio::commons::LogLevel::Normal);

    let rec = rerun::RecordingStreamBuilder::new("fastlio2_odom").save(out)?;

    // World origin marker for reference.
    rec.log_static(
        "world/origin",
        &rerun::Points3D::new([[0.0f32, 0.0, 0.0]])
            .with_radii([0.15])
            .with_colors([rerun::Color::from_rgb(255, 255, 255)]),
    )?;

    for (i, f) in files.iter().enumerate() {
        let arr: Array2<f64> = read_npy(f)?;
        if arr.nrows() == 0 || arr.ncols() < 4 {
            log::warn!("skip {} (shape {:?})", f, arr.dim());
            continue;
        }
        let traj: Vec<[f32; 3]> = (0..arr.nrows())
            .map(|r| [arr[[r, 1]] as f32, arr[[r, 2]] as f32, arr[[r, 3]] as f32])
            .collect();

        let (cr, cg, cb) = PALETTE[i % PALETTE.len()];
        let color = rerun::Color::from_rgb(cr, cg, cb);
        let base = format!("world/odom/run_{i:02}");

        rec.log_static(
            base.clone(),
            &rerun::LineStrips3D::new([traj.as_slice()])
                .with_colors([color])
                .with_radii([0.03]),
        )?;
        rec.log_static(
            format!("{base}/points"),
            &rerun::Points3D::new(traj)
                .with_radii([0.05])
                .with_colors([color]),
        )?;
        log::debug!("logged {} ({} points)", f, arr.nrows());
    }

    log::info!("wrote {out}");
    Ok(())
}
