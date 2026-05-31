//! PCL FFI bridge for VoxelGrid filter.
//!
//! When the `pcl` feature is enabled and PCL C++ libraries are available,
//! this module provides FFI bindings to PCL's VoxelGrid filter via `cxx`.
//! Without the feature, `downsample_pcl` falls back to the native Rust
//! voxel_grid implementation.
//!
//! To enable: `cargo build --features pcl`
//! Requires: PCL development headers, cxx-build in build.rs

use crate::commons::{Point, PointCloud};

// The cxx::bridge definition below is gated on the `pcl` feature.
// When enabled, a build.rs using cxx-build must compile the corresponding
// C++ shim (e.g., rust/cpp/pcl_ffi.cpp) that implements:
//
//   void voxel_grid_filter(
//       rust::Slice<const float> x, rust::Slice<const float> y, rust::Slice<const float> z,
//       float leaf_size,
//       rust::Vec<float>& out_x, rust::Vec<float>& out_y, rust::Vec<float>& out_z
//   );
//
// The shim would create a pcl::PointCloud<pcl::PointXYZ>, apply
// pcl::VoxelGrid, and write results back through the out vectors.

#[cfg(feature = "pcl")]
#[cxx::bridge(namespace = "pcl_ffi")]
mod ffi {
    unsafe extern "C++" {
        include!("pcl_ffi.h");

        fn voxel_grid_filter(
            x: &[f32],
            y: &[f32],
            z: &[f32],
            leaf_size: f32,
            out_x: &mut Vec<f32>,
            out_y: &mut Vec<f32>,
            out_z: &mut Vec<f32>,
        );
    }
}

pub fn downsample_pcl(cloud: &[Point], resolution: f64) -> PointCloud {
    #[cfg(feature = "pcl")]
    {
        let x: Vec<f32> = cloud.iter().map(|p| p.x).collect();
        let y: Vec<f32> = cloud.iter().map(|p| p.y).collect();
        let z: Vec<f32> = cloud.iter().map(|p| p.z).collect();
        let mut out_x = Vec::new();
        let mut out_y = Vec::new();
        let mut out_z = Vec::new();
        ffi::voxel_grid_filter(
            &x,
            &y,
            &z,
            resolution as f32,
            &mut out_x,
            &mut out_y,
            &mut out_z,
        );
        out_x
            .iter()
            .zip(out_y.iter())
            .zip(out_z.iter())
            .map(|((&px, &py), &pz)| Point {
                x: px,
                y: py,
                z: pz,
                intensity: 0.0,
                curvature: 0.0,
            })
            .collect()
    }
    #[cfg(not(feature = "pcl"))]
    {
        crate::voxel_grid::downsample(cloud, resolution)
    }
}
