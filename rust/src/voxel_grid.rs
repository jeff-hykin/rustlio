use std::collections::HashMap;
use crate::commons::Point;

pub fn downsample(cloud: &[Point], leaf_size: f64) -> Vec<Point> {
    if leaf_size <= 0.0 {
        return cloud.to_vec();
    }
    let inv = 1.0 / leaf_size;
    let mut grid: HashMap<(i64, i64, i64), (Point, f64)> = HashMap::new();

    for p in cloud {
        let key = (
            (p.x as f64 * inv).floor() as i64,
            (p.y as f64 * inv).floor() as i64,
            (p.z as f64 * inv).floor() as i64,
        );
        let mid_x = (key.0 as f64 + 0.5) * leaf_size;
        let mid_y = (key.1 as f64 + 0.5) * leaf_size;
        let mid_z = (key.2 as f64 + 0.5) * leaf_size;
        let dx = p.x as f64 - mid_x;
        let dy = p.y as f64 - mid_y;
        let dz = p.z as f64 - mid_z;
        let dist = dx * dx + dy * dy + dz * dz;

        let entry = grid.entry(key).or_insert((*p, f64::MAX));
        if dist < entry.1 {
            *entry = (*p, dist);
        }
    }

    grid.values().map(|(p, _)| *p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample_reduces_points() {
        let mut cloud = Vec::new();
        for i in 0..100 {
            cloud.push(Point {
                x: (i as f32) * 0.01,
                y: 0.0,
                z: 0.0,
                intensity: 1.0,
                curvature: 0.0,
            });
        }
        let result = downsample(&cloud, 0.1);
        assert!(result.len() < cloud.len());
        assert!(result.len() >= 10);
    }
}
