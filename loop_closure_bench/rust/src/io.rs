//! Neutral-file IO matching the C++ harness: clouds.bin + poses.tum, JSON out.
use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};
use std::fs::File;
use std::io::{BufReader, Read, Write};

pub struct Frame {
    pub ts: f64,
    pub pose: Isometry3<f64>,
    pub cloud: Vec<Vector3<f64>>, // body frame
}

/// Read TUM trajectory (`ts tx ty tz qx qy qz qw`) + the matching packed clouds.
/// clouds.bin: per frame `i32 n` then `n * 4 f32` (x,y,z,intensity); frame order
/// matches the TUM lines.
pub fn load(poses_path: &str, clouds_path: &str) -> Vec<Frame> {
    let poses_txt = std::fs::read_to_string(poses_path).expect("read poses");
    let mut cf = BufReader::new(File::open(clouds_path).expect("open clouds"));
    let mut frames = Vec::new();

    for line in poses_txt.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Vec<f64> = line.split_whitespace().map(|s| s.parse().unwrap()).collect();
        let (ts, tx, ty, tz, qx, qy, qz, qw) =
            (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
        let rot = UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz));
        let pose = Isometry3::from_parts(Translation3::new(tx, ty, tz), rot);

        let mut nbuf = [0u8; 4];
        cf.read_exact(&mut nbuf).expect("clouds.bin truncated");
        let n = i32::from_le_bytes(nbuf) as usize;
        let mut buf = vec![0u8; n * 4 * 4];
        cf.read_exact(&mut buf).expect("clouds.bin truncated");
        let mut cloud = Vec::with_capacity(n);
        for i in 0..n {
            let o = i * 16;
            let x = f32::from_le_bytes(buf[o..o + 4].try_into().unwrap()) as f64;
            let y = f32::from_le_bytes(buf[o + 4..o + 8].try_into().unwrap()) as f64;
            let z = f32::from_le_bytes(buf[o + 8..o + 12].try_into().unwrap()) as f64;
            cloud.push(Vector3::new(x, y, z));
        }
        frames.push(Frame { ts, pose, cloud });
    }
    frames
}

fn iso7(p: &Isometry3<f64>) -> String {
    let t = p.translation.vector;
    let q = p.rotation.quaternion();
    format!(
        "[{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}]",
        t.x, t.y, t.z, q.i, q.j, q.k, q.w
    )
}

pub struct OutKeyframe {
    pub ts: f64,
    pub raw: Isometry3<f64>,
    pub opt: Isometry3<f64>,
}

pub struct OutLoop {
    pub target: usize,
    pub source: usize,
    pub ts_target: f64,
    pub ts_source: f64,
    pub score: f64,
    pub offset_t: f64,
}

/// Write the same JSON schema the C++ harness emits (keyframes + loops).
pub fn write_json(path: &str, kfs: &[OutKeyframe], loops: &[OutLoop]) {
    let mut s = String::from("{\n  \"keyframes\": [\n");
    for (i, k) in kfs.iter().enumerate() {
        s += &format!(
            "    {{\"idx\":{},\"ts\":{:.9},\"raw\":{},\"opt\":{}}}{}\n",
            i,
            k.ts,
            iso7(&k.raw),
            iso7(&k.opt),
            if i + 1 < kfs.len() { "," } else { "" }
        );
    }
    s += "  ],\n  \"loops\": [\n";
    for (i, l) in loops.iter().enumerate() {
        s += &format!(
            "    {{\"target\":{},\"source\":{},\"ts_target\":{:.9},\"ts_source\":{:.9},\"score\":{:.6},\"offset_t\":{:.6}}}{}\n",
            l.target, l.source, l.ts_target, l.ts_source, l.score, l.offset_t,
            if i + 1 < loops.len() { "," } else { "" }
        );
    }
    s += "  ]\n}\n";
    File::create(path).expect("create out").write_all(s.as_bytes()).expect("write out");
}
