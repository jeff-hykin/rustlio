// Standalone benchmark driver for the C++ ICPLocalizer (no ROS2).
//
// Exercises the *real* relocalizer from FASTLIO2_ROS2/localizer
// (src/localizers/icp_localizer.{h,cpp}) — the two-stage (rough -> refine)
// point-to-point ICP — exactly as localizer_node.cpp's timerCB drives it:
//   m_localizer->setInput(scan);
//   m_localizer->align(initial_guess);
//
// This binary is the "C++ backend" for the relocalization benchmark. A future
// Rust relocalizer can implement the SAME CLI contract (read map + scans +
// trials, write results) and be compared on identical trials.
//
// CLI contract (stable — keep in sync with any Rust backend):
//   reloc_bench --map MAP.pcd --scans SCANS.bin --trials TRIALS.txt \
//               --out RESULTS.txt [cfgkey=val ...]
//
//   MAP.pcd      prior map, loaded via ICPLocalizer::loadMap (real code path)
//   SCANS.bin    concatenated body-frame clouds, each: [int32 n][n*(x,y,z,i) f32]
//                (same layout as the dataset's clouds.bin)
//   TRIALS.txt   one trial per line: "scan_idx tx ty tz qx qy qz qw"
//                (the 7 numbers are the SE3 initial guess: body->map)
//   RESULTS.txt  one line per trial: "converged tx ty tz qx qy qz qw time_ms"
//                pose = recovered body->map transform (== input guess if the
//                algorithm reports no convergence)
//
//   cfg overrides (else ICPConfig defaults): rough_scan_resolution,
//   rough_map_resolution, rough_max_iteration, rough_score_thresh,
//   refine_scan_resolution, refine_map_resolution, refine_max_iteration,
//   refine_score_thresh.

#include <Eigen/Eigen>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "localizers/commons.h"
#include "localizers/icp_localizer.h"

static std::vector<CloudType::Ptr> loadScans(const std::string &path) {
    std::ifstream cf(path, std::ios::binary);
    if (!cf) { std::cerr << "cannot open scans " << path << "\n"; std::exit(1); }
    std::vector<CloudType::Ptr> scans;
    while (true) {
        int32_t n = 0;
        cf.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
        if (!cf) break;  // clean EOF
        CloudType::Ptr cloud(new CloudType);
        cloud->resize(n);
        std::vector<float> buf(static_cast<size_t>(n) * 4);
        cf.read(reinterpret_cast<char *>(buf.data()),
                static_cast<std::streamsize>(buf.size() * sizeof(float)));
        if (!cf) { std::cerr << "scans.bin truncated at scan " << scans.size() << "\n"; std::exit(1); }
        for (int32_t i = 0; i < n; ++i) {
            PointType &p = cloud->points[i];
            p.x = buf[i * 4 + 0];
            p.y = buf[i * 4 + 1];
            p.z = buf[i * 4 + 2];
            p.intensity = buf[i * 4 + 3];
        }
        cloud->width = n;
        cloud->height = 1;
        scans.push_back(cloud);
    }
    return scans;
}

int main(int argc, char **argv) {
    std::string map_path, scans_path, trials_path, out_path;
    ICPConfig cfg;  // defaults from icp_localizer.h
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() { return std::string(argv[++i]); };
        if (a == "--map") map_path = need();
        else if (a == "--scans") scans_path = need();
        else if (a == "--trials") trials_path = need();
        else if (a == "--out") out_path = need();
        else {
            auto eq = a.find('=');
            if (eq != std::string::npos) kv[a.substr(0, eq)] = a.substr(eq + 1);
        }
    }
    auto getd = [&](const char *k, double d) { return kv.count(k) ? std::stod(kv[k]) : d; };
    auto geti = [&](const char *k, int d) { return kv.count(k) ? std::stoi(kv[k]) : d; };
    cfg.rough_scan_resolution = getd("rough_scan_resolution", cfg.rough_scan_resolution);
    cfg.rough_map_resolution = getd("rough_map_resolution", cfg.rough_map_resolution);
    cfg.rough_max_iteration = geti("rough_max_iteration", cfg.rough_max_iteration);
    cfg.rough_score_thresh = getd("rough_score_thresh", cfg.rough_score_thresh);
    cfg.refine_scan_resolution = getd("refine_scan_resolution", cfg.refine_scan_resolution);
    cfg.refine_map_resolution = getd("refine_map_resolution", cfg.refine_map_resolution);
    cfg.refine_max_iteration = geti("refine_max_iteration", cfg.refine_max_iteration);
    cfg.refine_score_thresh = getd("refine_score_thresh", cfg.refine_score_thresh);

    if (map_path.empty() || scans_path.empty() || trials_path.empty() || out_path.empty()) {
        std::cerr << "usage: reloc_bench --map M.pcd --scans S.bin --trials T.txt --out R.txt [k=v ...]\n";
        return 1;
    }

    ICPLocalizer loc(cfg);
    if (!loc.loadMap(map_path)) { std::cerr << "loadMap failed: " << map_path << "\n"; return 1; }
    std::cerr << "[reloc_bench] map rough=" << loc.roughMap()->size()
              << " refine=" << loc.refineMap()->size() << " points\n";

    std::vector<CloudType::Ptr> scans = loadScans(scans_path);
    std::cerr << "[reloc_bench] loaded " << scans.size() << " scans\n";

    std::ifstream tf(trials_path);
    if (!tf) { std::cerr << "cannot open trials " << trials_path << "\n"; return 1; }
    std::ofstream os(out_path);
    os.precision(9);

    std::string line;
    size_t n_trials = 0, n_conv = 0;
    while (std::getline(tf, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        long idx;
        double tx, ty, tz, qx, qy, qz, qw;
        if (!(ss >> idx >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;
        if (idx < 0 || idx >= static_cast<long>(scans.size())) {
            std::cerr << "trial scan_idx out of range: " << idx << "\n"; return 1;
        }
        M4F guess = M4F::Identity();
        Eigen::Quaternionf q(static_cast<float>(qw), static_cast<float>(qx),
                             static_cast<float>(qy), static_cast<float>(qz));
        q.normalize();
        guess.block<3, 3>(0, 0) = q.toRotationMatrix();
        guess.block<3, 1>(0, 3) = V3F(static_cast<float>(tx), static_cast<float>(ty),
                                      static_cast<float>(tz));

        loc.setInput(scans[idx]);
        auto t0 = std::chrono::steady_clock::now();
        bool conv = loc.align(guess);  // modifies `guess` to recovered pose on success
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        Eigen::Matrix3f R = guess.block<3, 3>(0, 0);
        Eigen::Quaternionf rq(R);
        rq.normalize();
        V3F t = guess.block<3, 1>(0, 3);
        os << (conv ? 1 : 0) << " " << t.x() << " " << t.y() << " " << t.z() << " "
           << rq.x() << " " << rq.y() << " " << rq.z() << " " << rq.w() << " " << ms << "\n";
        n_trials++;
        if (conv) n_conv++;
    }
    std::cerr << "[reloc_bench] trials=" << n_trials << " converged=" << n_conv << "\n";
    std::cerr << "[reloc_bench] wrote " << out_path << "\n";
    return 0;
}
