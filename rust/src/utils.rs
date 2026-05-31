use crate::commons::Point;

pub fn stamp_to_sec(sec: u32, nsec: u32) -> f64 {
    sec as f64 + nsec as f64 * 1e-9
}

pub fn sec_to_stamp(t: f64) -> (u32, u32) {
    let sec = t as u32;
    let nsec = ((t - sec as f64) * 1e9) as u32;
    (sec, nsec)
}

pub fn livox_point_valid(tag: u8, line: u8) -> bool {
    line < 4 && ((tag & 0x30) == 0x10 || (tag & 0x30) == 0x00)
}

pub fn livox_to_point(
    x: f32,
    y: f32,
    z: f32,
    reflectivity: u8,
    offset_time_ns: u32,
    min_range: f64,
    max_range: f64,
) -> Option<Point> {
    let r2 = x * x + y * y + z * z;
    let min_r2 = (min_range * min_range) as f32;
    let max_r2 = (max_range * max_range) as f32;
    if r2 < min_r2 || r2 > max_r2 {
        return None;
    }
    Some(Point {
        x,
        y,
        z,
        intensity: reflectivity as f32,
        curvature: offset_time_ns as f32 / 1_000_000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stamp_roundtrip() {
        let t = 1717200000.123456789;
        let (sec, nsec) = sec_to_stamp(t);
        let recovered = stamp_to_sec(sec, nsec);
        assert!((recovered - t).abs() < 1e-6);
    }

    #[test]
    fn test_livox_filter() {
        assert!(livox_point_valid(0x00, 0));
        assert!(livox_point_valid(0x10, 3));
        assert!(!livox_point_valid(0x30, 0));
        assert!(!livox_point_valid(0x00, 4));
    }

    #[test]
    fn test_livox_to_point_range() {
        assert!(livox_to_point(1.0, 0.0, 0.0, 100, 5000, 0.5, 20.0).is_some());
        assert!(livox_to_point(0.1, 0.0, 0.0, 100, 5000, 0.5, 20.0).is_none());
        assert!(livox_to_point(25.0, 0.0, 0.0, 100, 5000, 0.5, 20.0).is_none());
    }
}
