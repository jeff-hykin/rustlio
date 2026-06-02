// Standalone harness for the in-repo C++ FAST-LIO map_builder (the source the
// Rust rustlio2 was ported from). Reads the flat sensor dump produced by
// faithful_check/dump_flat.py (identical bytes to the Rust flat-runner), drives
// MapBuilder exactly like lio_node's syncPackage loop, and logs per-frame state
// so the two implementations can be diffed to find where they first diverge.
//
// No ROS2: links map_builder/*.cpp directly (PCL + Eigen + Sophus).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include "map_builder/map_builder.h"
#include "map_builder/commons.h"
#include "map_builder/ieskf.h"

struct ImuRec { double t, ax, ay, az, gx, gy, gz; };
struct Frame { double t_start; std::vector<std::array<float, 5>> pts; }; // x,y,z,intensity,curv_ms

static bool read_flat(const char *path, std::vector<ImuRec> &imus, std::vector<Frame> &frames) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return false; }
    char magic[4]; if (fread(magic, 1, 4, f) != 4 || std::string(magic, 4) != "FLT1") { printf("bad magic\n"); return false; }
    uint64_t n_imu, n_frames;
    if (fread(&n_imu, 8, 1, f) != 1 || fread(&n_frames, 8, 1, f) != 1) return false;
    imus.resize(n_imu);
    for (uint64_t i = 0; i < n_imu; i++) { double r[7]; if (fread(r, 8, 7, f) != 7) return false;
        imus[i] = {r[0], r[1], r[2], r[3], r[4], r[5], r[6]}; }
    frames.resize(n_frames);
    for (uint64_t i = 0; i < n_frames; i++) {
        double ts; uint64_t np;
        if (fread(&ts, 8, 1, f) != 1 || fread(&np, 8, 1, f) != 1) return false;
        frames[i].t_start = ts; frames[i].pts.resize(np);
        for (uint64_t k = 0; k < np; k++) { float p[5]; if (fread(p, 4, 5, f) != 5) return false;
            frames[i].pts[k] = {p[0], p[1], p[2], p[3], p[4]}; }
    }
    fclose(f);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 3) { printf("usage: ref_run <flat> <out_csv>\n"); return 2; }
    std::vector<ImuRec> imus; std::vector<Frame> frames;
    if (!read_flat(argv[1], imus, frames)) return 1;
    printf("loaded %zu imu, %zu frames\n", imus.size(), frames.size());

    // Effective config = config_examples/mid360.yaml resolved by rustlio2.
    Config cfg;
    cfg.lidar_filter_num = 3;
    cfg.lidar_min_range = 0.5; cfg.lidar_max_range = 20.0;
    cfg.scan_resolution = 0.15; cfg.map_resolution = 0.3;
    cfg.cube_len = 300; cfg.det_range = 100; cfg.move_thresh = 1.5;
    cfg.na = 0.1; cfg.ng = 0.1; cfg.nba = 0.0001; cfg.nbg = 0.0001;
    cfg.imu_init_num = 20; cfg.near_search_num = 5; cfg.ieskf_max_iter = 5;
    cfg.gravity_align = true; cfg.esti_il = false;
    cfg.r_il = M3D::Identity();
    cfg.t_il = V3D(-0.011, -0.02329, 0.04412);
    cfg.lidar_cov_inv = 1000.0;

    auto kf = std::make_shared<IESKF>();
    auto builder = std::make_shared<MapBuilder>(cfg, kf);

    FILE *out = fopen(argv[2], "w");
    fprintf(out, "t,x,y,z,vx,vy,vz,gx,gy,gz,bax,bay,baz\n");

    size_t imu_idx = 0;
    double first_t = -1;
    long n_out = 0;
    for (auto &fr : frames) {
        // Build cloud, sort ascending by curvature (lio_node syncPackage).
        CloudType::Ptr cloud(new CloudType);
        cloud->reserve(fr.pts.size());
        for (auto &p : fr.pts) {
            PointType pt; pt.x = p[0]; pt.y = p[1]; pt.z = p[2];
            pt.intensity = p[3]; pt.curvature = p[4];
            cloud->push_back(pt);
        }
        std::sort(cloud->points.begin(), cloud->points.end(),
                  [](const PointType &a, const PointType &b) { return a.curvature < b.curvature; });
        if (cloud->points.empty()) continue;
        double cloud_end = fr.t_start + cloud->points.back().curvature / 1000.0;

        SyncPackage pkg;
        pkg.cloud = cloud;
        pkg.cloud_start_time = fr.t_start;
        pkg.cloud_end_time = cloud_end;
        while (imu_idx < imus.size() && imus[imu_idx].t < cloud_end) {
            ImuRec &r = imus[imu_idx];
            pkg.imus.emplace_back(V3D(r.ax, r.ay, r.az), V3D(r.gx, r.gy, r.gz), r.t);
            imu_idx++;
        }
        if (pkg.imus.empty()) continue;

        builder->process(pkg);
        if (builder->status() != MAPPING) continue;
        const State &x = kf->x();
        if (first_t < 0) first_t = fr.t_start;
        fprintf(out, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                fr.t_start, x.t_wi.x(), x.t_wi.y(), x.t_wi.z(),
                x.v.x(), x.v.y(), x.v.z(), x.g.x(), x.g.y(), x.g.z(),
                x.ba.x(), x.ba.y(), x.ba.z());
        n_out++;
    }
    fclose(out);
    printf("wrote %ld mapping frames to %s\n", n_out, argv[2]);
    return 0;
}
