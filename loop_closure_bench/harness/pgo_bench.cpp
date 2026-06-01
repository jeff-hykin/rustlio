// Standalone benchmark driver for the C++ SimplePGO (no ROS2).
//
// Reads the neutral files produced by export_dataset.py (+ a poses file that may
// carry injected drift), feeds them frame-by-frame through SimplePGO exactly as
// pgo_node.cpp's timerCB does, and writes the resulting pose graph (raw +
// optimized keyframe poses) and detected loop edges as JSON.
//
//   pgo_bench --clouds clouds.bin --poses poses.tum --out out.json [key=val ...]
//
// Config keys (override SimplePGO Config defaults): key_pose_delta_deg,
// key_pose_delta_trans, loop_search_radius, loop_time_tresh, loop_score_tresh,
// loop_submap_half_range, submap_resolution, min_loop_detect_duration.

#include <Eigen/Eigen>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "pgos/commons.h"
#include "pgos/simple_pgo.h"
#include "ivan_pgo.h"

struct Frame {
    double ts;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
    CloudType::Ptr cloud;
};

static std::vector<Frame> load(const std::string &poses_path, const std::string &clouds_path) {
    std::vector<Frame> frames;
    std::ifstream pf(poses_path);
    if (!pf) { std::cerr << "cannot open poses " << poses_path << "\n"; std::exit(1); }
    std::ifstream cf(clouds_path, std::ios::binary);
    if (!cf) { std::cerr << "cannot open clouds " << clouds_path << "\n"; std::exit(1); }

    std::string line;
    while (std::getline(pf, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        Frame fr;
        double tx, ty, tz, qx, qy, qz, qw;
        ss >> fr.ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        fr.t = Eigen::Vector3d(tx, ty, tz);
        fr.q = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
        // matching cloud
        int32_t n = 0;
        cf.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
        if (!cf) { std::cerr << "clouds.bin ran out at frame " << frames.size() << "\n"; std::exit(1); }
        fr.cloud = CloudType::Ptr(new CloudType);
        fr.cloud->resize(n);
        std::vector<float> buf(static_cast<size_t>(n) * 4);
        cf.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(float)));
        for (int32_t i = 0; i < n; ++i) {
            PointType &p = fr.cloud->points[i];
            p.x = buf[i * 4 + 0];
            p.y = buf[i * 4 + 1];
            p.z = buf[i * 4 + 2];
            p.intensity = buf[i * 4 + 3];
        }
        fr.cloud->width = n;
        fr.cloud->height = 1;
        frames.push_back(std::move(fr));
    }
    return frames;
}

static void w7(std::ostream &os, const Eigen::Vector3d &t, const Eigen::Quaterniond &q) {
    os << "[" << t.x() << "," << t.y() << "," << t.z() << "," << q.x() << "," << q.y() << "," << q.z()
       << "," << q.w() << "]";
}

// Drive any PGO with the shared interface (addKeyPose / searchForLoopPairs /
// cachePairs / smoothAndUpdate / keyPoses) frame-by-frame, as pgo_node.cpp does,
// and write keyframes + loop edges as JSON.
template <class PGO>
static int run(const Config &cfg, const std::vector<Frame> &frames, const std::string &out) {
    PGO pgo(cfg);
    struct LoopRec {
        size_t target, source;
        double ts_t, ts_s, score, dt_off;  // dt_off = ICP offset translation norm
    };
    std::vector<LoopRec> loops;

    for (const Frame &fr : frames) {
        CloudWithPose cp;
        int64_t sec = static_cast<int64_t>(fr.ts);
        cp.pose.setTime(static_cast<int32_t>(sec), static_cast<uint32_t>((fr.ts - sec) * 1e9));
        cp.pose.r = fr.q.toRotationMatrix();
        cp.pose.t = fr.t;
        cp.cloud = fr.cloud;
        if (!pgo.addKeyPose(cp)) continue;
        pgo.searchForLoopPairs();
        auto &kps = pgo.keyPoses();
        for (const LoopPair &lp : pgo.cachePairs()) {
            loops.push_back({lp.target_id, lp.source_id, kps[lp.target_id].time,
                             kps[lp.source_id].time, lp.score, lp.t_offset.norm()});
        }
        pgo.smoothAndUpdate();
    }

    auto &kps = pgo.keyPoses();
    std::cerr << "[pgo_bench] keyframes=" << kps.size() << " loops=" << loops.size() << "\n";

    std::ofstream os(out);
    os.precision(16);  // timestamps are ~1.78e9; need full precision for sub-second part
    os << "{\n  \"keyframes\": [\n";
    for (size_t i = 0; i < kps.size(); ++i) {
        os << "    {\"idx\":" << i << ",\"ts\":" << kps[i].time << ",\"raw\":";
        w7(os, kps[i].t_local, Eigen::Quaterniond(kps[i].r_local));
        os << ",\"opt\":";
        w7(os, kps[i].t_global, Eigen::Quaterniond(kps[i].r_global));
        os << "}" << (i + 1 < kps.size() ? "," : "") << "\n";
    }
    os << "  ],\n  \"loops\": [\n";
    for (size_t i = 0; i < loops.size(); ++i) {
        os << "    {\"target\":" << loops[i].target << ",\"source\":" << loops[i].source
           << ",\"ts_target\":" << loops[i].ts_t << ",\"ts_source\":" << loops[i].ts_s
           << ",\"score\":" << loops[i].score << ",\"offset_t\":" << loops[i].dt_off << "}"
           << (i + 1 < loops.size() ? "," : "") << "\n";
    }
    os << "  ]\n}\n";
    std::cerr << "[pgo_bench] wrote " << out << "\n";
    return 0;
}

int main(int argc, char **argv) {
    std::string clouds, poses, out;
    Config cfg;  // defaults from simple_pgo.h
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char *) { return std::string(argv[++i]); };
        if (a == "--clouds") clouds = need("");
        else if (a == "--poses") poses = need("");
        else if (a == "--out") out = need("");
        else {
            auto eq = a.find('=');
            if (eq != std::string::npos) kv[a.substr(0, eq)] = a.substr(eq + 1);
        }
    }
    auto getd = [&](const char *k, double d) { return kv.count(k) ? std::stod(kv[k]) : d; };
    auto geti = [&](const char *k, int d) { return kv.count(k) ? std::stoi(kv[k]) : d; };
    cfg.key_pose_delta_deg = getd("key_pose_delta_deg", cfg.key_pose_delta_deg);
    cfg.key_pose_delta_trans = getd("key_pose_delta_trans", cfg.key_pose_delta_trans);
    cfg.loop_search_radius = getd("loop_search_radius", cfg.loop_search_radius);
    cfg.loop_time_tresh = getd("loop_time_tresh", cfg.loop_time_tresh);
    cfg.loop_score_tresh = getd("loop_score_tresh", cfg.loop_score_tresh);
    cfg.loop_submap_half_range = geti("loop_submap_half_range", cfg.loop_submap_half_range);
    cfg.submap_resolution = getd("submap_resolution", cfg.submap_resolution);
    cfg.min_loop_detect_duration = getd("min_loop_detect_duration", cfg.min_loop_detect_duration);
    cfg.max_icp_correspondence_dist = getd("max_icp_correspondence_dist", cfg.max_icp_correspondence_dist);
    cfg.max_loop_offset = getd("max_loop_offset", cfg.max_loop_offset);
    cfg.loop_source_submap_half_range = geti("loop_source_submap_half_range", cfg.loop_source_submap_half_range);

    if (clouds.empty() || poses.empty() || out.empty()) {
        std::cerr << "usage: pgo_bench --clouds f --poses f --out f [key=val ...]\n";
        return 1;
    }

    std::vector<Frame> frames = load(poses, clouds);
    std::cerr << "[pgo_bench] loaded " << frames.size() << " frames\n";

    std::string impl = kv.count("impl") ? kv["impl"] : "stock";
    if (impl == "ivan")
        return run<IvanPGO>(cfg, frames, out);
    return run<SimplePGO>(cfg, frames, out);
}
