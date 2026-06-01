// C++ port of the global FPFH+RANSAC relocalizer (dimos
// dimos/mapping/relocalization/relocalize.py). A GLOBAL method: given a prior
// map and a local submap, recover the submap's pose in the map frame with NO
// initial guess.
//
// Pipeline (faithful to relocalize.py):
//   multi-scale FPFH + RANSAC feature matching (several restarts per scale)
//   -> add 180-deg yaw-flip variants (rotate about the submap xy-centroid)
//   -> gravity filter (reject z-tilt > 10 deg)
//   -> rerank candidates by WALL-ONLY fine-scale inlier fitness, keep top-K
//   -> point-to-plane ICP polish on walls, pick best
//   -> final point-to-plane ICP on full fine clouds
//
// Same CLI contract as reloc_bench so it plugs into bench_reloc.py unchanged;
// the per-trial initial guess is IGNORED (this is a global method). A trial's
// "scan" is a local submap (built by gen_scenarios for the global scenario).
//
//   global_reloc_bench --map M.pcd --scans S.bin --trials T.txt --out R.txt [k=v ...]
//   RESULTS line: "converged tx ty tz qx qy qz qw time_ms"  (converged = fitness gate)
//
// cfg overrides: ransac_iters, restarts_fine, restarts_coarse, accept_fitness.

#include <Eigen/Eigen>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/icp.h>

using XYZ = pcl::PointXYZ;
using Cloud = pcl::PointCloud<XYZ>;
using Normals = pcl::PointCloud<pcl::Normal>;
using PN = pcl::PointNormal;
using CloudN = pcl::PointCloud<PN>;
using FPFH = pcl::PointCloud<pcl::FPFHSignature33>;

// scales: (voxel, restarts). Matches relocalize.py SCALE_PLAN (0.2:8, 0.3:8, 0.8:1).
struct Scale { double vs; int restarts; };
static const double FINE_VOXEL = 0.1;
static const double RERANK_DIST = FINE_VOXEL * 1.5;
static const double GRAVITY_TILT_MAX_DEG = 10.0;
static const int TOP_K = 10;

static int g_ransac_iters = 100000;  // PCL SACPrerejective is fixed-iteration

// Hashed centroid voxel downsample (like Open3D/numpy). PCL's VoxelGrid
// overflows its int voxel index on large outdoor maps (extent/leaf)^3 > 2^31
// and silently returns the cloud un-downsampled; hashing avoids that.
static Cloud::Ptr downsample(const Cloud::Ptr &in, double vs) {
    Cloud::Ptr out(new Cloud);
    if (vs <= 0 || in->empty()) { *out = *in; return out; }
    const double inv = 1.0 / vs;
    std::unordered_map<int64_t, std::pair<Eigen::Vector3d, int>> cells;
    cells.reserve(in->size());
    auto key = [](int x, int y, int z) -> int64_t {
        // 21 bits per axis, offset to stay positive (range +/- ~1e6 voxels).
        return (int64_t)(x + 1048576) | ((int64_t)(y + 1048576) << 21) |
               ((int64_t)(z + 1048576) << 42);
    };
    for (const auto &p : in->points) {
        int ix = (int)std::floor(p.x * inv), iy = (int)std::floor(p.y * inv),
            iz = (int)std::floor(p.z * inv);
        auto &c = cells[key(ix, iy, iz)];
        c.first += Eigen::Vector3d(p.x, p.y, p.z);
        c.second++;
    }
    out->reserve(cells.size());
    for (const auto &kv : cells) {
        Eigen::Vector3d m = kv.second.first / kv.second.second;
        out->push_back(XYZ(m.x(), m.y(), m.z()));
    }
    out->width = out->size();
    out->height = 1;
    return out;
}

static Normals::Ptr estimateNormals(const Cloud::Ptr &c, double radius) {
    Normals::Ptr n(new Normals);
    pcl::NormalEstimationOMP<XYZ, pcl::Normal> ne;
    pcl::search::KdTree<XYZ>::Ptr tree(new pcl::search::KdTree<XYZ>);
    ne.setInputCloud(c);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(radius);
    ne.compute(*n);
    return n;
}

static FPFH::Ptr computeFPFH(const Cloud::Ptr &c, const Normals::Ptr &n, double radius) {
    FPFH::Ptr f(new FPFH);
    pcl::FPFHEstimationOMP<XYZ, pcl::Normal, pcl::FPFHSignature33> fe;
    pcl::search::KdTree<XYZ>::Ptr tree(new pcl::search::KdTree<XYZ>);
    fe.setInputCloud(c);
    fe.setInputNormals(n);
    fe.setSearchMethod(tree);
    fe.setRadiusSearch(radius);
    fe.compute(*f);
    return f;
}

static CloudN::Ptr makePN(const Cloud::Ptr &c, const Normals::Ptr &n) {
    CloudN::Ptr out(new CloudN);
    pcl::concatenateFields(*c, *n, *out);
    return out;
}

// Keep roughly-vertical surfaces (walls): |normal_z| < 0.7. Floor/ceiling fit
// any yaw, so they hide wall misalignment in scoring.
static CloudN::Ptr wallSubset(const CloudN::Ptr &c) {
    CloudN::Ptr out(new CloudN);
    out->reserve(c->size());
    for (const auto &p : c->points)
        if (std::abs(p.normal_z) < 0.7) out->push_back(p);
    if (out->size() < 100) return c;  // too sparse -> fall back to full cloud
    out->width = out->size();
    out->height = 1;
    return out;
}

// Fraction of src points (under T) with a tgt neighbour within `dist`. This is
// Open3D's evaluate_registration().fitness (inlier ratio).
static double inlierFitness(const CloudN::Ptr &src, const pcl::KdTreeFLANN<XYZ> &tgt_kd,
                            const Eigen::Matrix4f &T, double dist) {
    if (src->empty()) return 0.0;
    std::vector<int> idx(1);
    std::vector<float> d2(1);
    double d2thresh = dist * dist;
    size_t inl = 0;
    for (const auto &p : src->points) {
        Eigen::Vector4f q = T * Eigen::Vector4f(p.x, p.y, p.z, 1.0f);
        XYZ s;
        s.x = q.x(); s.y = q.y(); s.z = q.z();
        if (tgt_kd.nearestKSearch(s, 1, idx, d2) > 0 && d2[0] <= d2thresh) inl++;
    }
    return static_cast<double>(inl) / src->size();
}

static double gravityTiltDeg(const Eigen::Matrix4f &T) {
    Eigen::Vector3f z = T.block<3, 3>(0, 0) * Eigen::Vector3f(0, 0, 1);
    return std::acos(std::min(1.0f, std::max(-1.0f, z.z()))) * 180.0 / M_PI;
}

static Eigen::Matrix4f pointToPlaneICP(const CloudN::Ptr &src, const CloudN::Ptr &tgt,
                                       const Eigen::Matrix4f &init, double maxCorr, int iters) {
    pcl::IterativeClosestPointWithNormals<PN, PN> icp;
    icp.setInputSource(src);
    icp.setInputTarget(tgt);
    icp.setMaxCorrespondenceDistance(maxCorr);
    icp.setMaximumIterations(iters);
    CloudN out;
    icp.align(out, init);
    return icp.hasConverged() ? icp.getFinalTransformation() : init;
}

// --- target (global map) caches, computed once ---
struct TargetCache {
    std::vector<Scale> scales;
    std::vector<Cloud::Ptr> down;       // per scale
    std::vector<FPFH::Ptr> fpfh;        // per scale
    std::vector<pcl::SampleConsensusPrerejective<XYZ, XYZ, pcl::FPFHSignature33>> sac;  // per scale
    CloudN::Ptr fine;                   // 0.1 voxel + normals
    CloudN::Ptr walls;                  // wall subset of fine
    pcl::KdTreeFLANN<XYZ> walls_kd;     // for rerank fitness
};

static void buildTarget(const Cloud::Ptr &map, TargetCache &tc) {
    for (const Scale &s : tc.scales) {
        Cloud::Ptr d = downsample(map, s.vs);
        Normals::Ptr n = estimateNormals(d, s.vs * 2.0);
        FPFH::Ptr f = computeFPFH(d, n, s.vs * 5.0);
        tc.down.push_back(d);
        tc.fpfh.push_back(f);
    }
    Cloud::Ptr fineD = downsample(map, FINE_VOXEL);
    Normals::Ptr fineN = estimateNormals(fineD, FINE_VOXEL * 2.0);
    tc.fine = makePN(fineD, fineN);
    tc.walls = wallSubset(tc.fine);
    Cloud::Ptr wxyz(new Cloud);
    pcl::copyPointCloud(*tc.walls, *wxyz);
    tc.walls_kd.setInputCloud(wxyz);

    // Persistent SAC per scale with target set once (avoids rebuilding per query).
    tc.sac.resize(tc.scales.size());
    for (size_t i = 0; i < tc.scales.size(); ++i) {
        double dist = tc.scales[i].vs * 1.5;
        auto &sac = tc.sac[i];
        sac.setInputTarget(tc.down[i]);
        sac.setTargetFeatures(tc.fpfh[i]);
        sac.setMaximumIterations(g_ransac_iters);
        sac.setNumberOfSamples(3);
        sac.setCorrespondenceRandomness(5);
        sac.setSimilarityThreshold(0.9);            // CorrespondenceCheckerBasedOnEdgeLength(0.9)
        sac.setMaxCorrespondenceDistance(dist);     // CheckerBasedOnDistance + RANSAC corr dist
        sac.setInlierFraction(0.25);
    }
}

struct RelocResult { Eigen::Matrix4f T; double fitness; };

static RelocResult relocalize(const Cloud::Ptr &local, TargetCache &tc) {
    Cloud::Ptr srcFineD = downsample(local, FINE_VOXEL);
    Normals::Ptr srcFineN = estimateNormals(srcFineD, FINE_VOXEL * 2.0);
    CloudN::Ptr srcFine = makePN(srcFineD, srcFineN);
    CloudN::Ptr srcWalls = wallSubset(srcFine);

    std::vector<Eigen::Matrix4f> candidates;
    for (size_t i = 0; i < tc.scales.size(); ++i) {
        const Scale &s = tc.scales[i];
        Cloud::Ptr d = downsample(local, s.vs);
        Normals::Ptr n = estimateNormals(d, s.vs * 2.0);
        FPFH::Ptr f = computeFPFH(d, n, s.vs * 5.0);
        for (int r = 0; r < s.restarts; ++r) {
            tc.sac[i].setInputSource(d);
            tc.sac[i].setSourceFeatures(f);
            Cloud out;
            tc.sac[i].align(out);
            candidates.push_back(tc.sac[i].getFinalTransformation());
        }
    }

    // 180-deg yaw flip about the submap xy-centroid ("same place, opposite heading").
    Eigen::Vector4f c;
    pcl::compute3DCentroid(*srcFine, c);
    Eigen::Matrix4f flip = Eigen::Matrix4f::Identity();
    flip(0, 0) = -1; flip(1, 1) = -1;
    flip(0, 3) = 2.0f * c.x(); flip(1, 3) = 2.0f * c.y();
    size_t base = candidates.size();
    for (size_t i = 0; i < base; ++i) candidates.push_back(candidates[i] * flip);

    // Gravity filter (fall back to all if everything is tilted).
    std::vector<Eigen::Matrix4f> pool;
    for (const auto &T : candidates)
        if (gravityTiltDeg(T) <= GRAVITY_TILT_MAX_DEG) pool.push_back(T);
    if (pool.empty()) pool = candidates;

    // Rerank by wall-only fine fitness, keep top-K.
    std::vector<std::pair<double, Eigen::Matrix4f>> scored;
    for (const auto &T : pool)
        scored.emplace_back(inlierFitness(srcWalls, tc.walls_kd, T, RERANK_DIST), T);
    std::sort(scored.begin(), scored.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    if ((int)scored.size() > TOP_K) scored.resize(TOP_K);

    // Polish each on walls (point-to-plane), pick best wall fitness.
    double bestFit = -1.0;
    Eigen::Matrix4f bestT = Eigen::Matrix4f::Identity();
    for (const auto &sc : scored) {
        Eigen::Matrix4f T = pointToPlaneICP(srcWalls, tc.walls, sc.second, RERANK_DIST, 70);
        double fit = inlierFitness(srcWalls, tc.walls_kd, T, RERANK_DIST);
        if (fit > bestFit) { bestFit = fit; bestT = T; }
    }

    // Final ICP on full fine clouds.
    Eigen::Matrix4f finalT = pointToPlaneICP(srcFine, tc.fine, bestT, RERANK_DIST, 50);
    return {finalT, bestFit};
}

static std::vector<Cloud::Ptr> loadScans(const std::string &path) {
    std::ifstream cf(path, std::ios::binary);
    if (!cf) { std::cerr << "cannot open scans " << path << "\n"; std::exit(1); }
    std::vector<Cloud::Ptr> scans;
    while (true) {
        int32_t n = 0;
        cf.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
        if (!cf) break;
        Cloud::Ptr cloud(new Cloud);
        cloud->resize(n);
        std::vector<float> buf(static_cast<size_t>(n) * 4);
        cf.read(reinterpret_cast<char *>(buf.data()),
                static_cast<std::streamsize>(buf.size() * sizeof(float)));
        for (int32_t i = 0; i < n; ++i) {
            (*cloud)[i].x = buf[i * 4 + 0];
            (*cloud)[i].y = buf[i * 4 + 1];
            (*cloud)[i].z = buf[i * 4 + 2];
        }
        cloud->width = n; cloud->height = 1;
        scans.push_back(cloud);
    }
    return scans;
}

int main(int argc, char **argv) {
    std::string map_path, scans_path, trials_path, out_path;
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() { return std::string(argv[++i]); };
        if (a == "--map") map_path = need();
        else if (a == "--scans") scans_path = need();
        else if (a == "--trials") trials_path = need();
        else if (a == "--out") out_path = need();
        else { auto eq = a.find('='); if (eq != std::string::npos) kv[a.substr(0, eq)] = a.substr(eq + 1); }
    }
    auto getd = [&](const char *k, double d) { return kv.count(k) ? std::stod(kv[k]) : d; };
    auto geti = [&](const char *k, int d) { return kv.count(k) ? std::stoi(kv[k]) : d; };
    g_ransac_iters = geti("ransac_iters", g_ransac_iters);
    int restarts_fine = geti("restarts_fine", 8);
    int restarts_coarse = geti("restarts_coarse", 1);
    double accept_fitness = getd("accept_fitness", 0.15);

    if (map_path.empty() || scans_path.empty() || trials_path.empty() || out_path.empty()) {
        std::cerr << "usage: global_reloc_bench --map M.pcd --scans S.bin --trials T.txt --out R.txt [k=v]\n";
        return 1;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr rawmap(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile(map_path, *rawmap) < 0) { std::cerr << "loadPCD failed: " << map_path << "\n"; return 1; }
    Cloud::Ptr map(new Cloud);
    pcl::copyPointCloud(*rawmap, *map);

    TargetCache tc;
    tc.scales = {{0.2, restarts_fine}, {0.3, restarts_fine}, {0.8, restarts_coarse}};
    std::cerr << "[global_reloc_bench] building target from " << map->size() << " map points...\n";
    auto tb = std::chrono::steady_clock::now();
    buildTarget(map, tc);
    std::cerr << "[global_reloc_bench] target ready (fine=" << tc.fine->size()
              << " walls=" << tc.walls->size() << ") in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count() << "s\n";

    std::vector<Cloud::Ptr> scans = loadScans(scans_path);
    std::cerr << "[global_reloc_bench] loaded " << scans.size() << " submaps\n";

    std::ifstream tf(trials_path);
    if (!tf) { std::cerr << "cannot open trials " << trials_path << "\n"; return 1; }
    std::ofstream os(out_path);
    os.precision(9);

    std::string line;
    size_t n = 0, conv = 0;
    while (std::getline(tf, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        long idx; double tmp;
        if (!(ss >> idx)) continue;
        for (int k = 0; k < 7; ++k) ss >> tmp;  // ignore the initial guess
        if (idx < 0 || idx >= (long)scans.size()) { std::cerr << "bad idx " << idx << "\n"; return 1; }

        auto t0 = std::chrono::steady_clock::now();
        RelocResult r = relocalize(scans[idx], tc);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        Eigen::Matrix3f R = r.T.block<3, 3>(0, 0);
        Eigen::Quaternionf q(R); q.normalize();
        Eigen::Vector3f t = r.T.block<3, 1>(0, 3);
        bool ok = r.fitness >= accept_fitness;
        os << (ok ? 1 : 0) << " " << t.x() << " " << t.y() << " " << t.z() << " "
           << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " " << ms << "\n";
        n++; if (ok) conv++;
        std::cerr << "  trial " << n << "/idx" << idx << " fitness=" << r.fitness
                  << " " << ms << "ms\n";
    }
    std::cerr << "[global_reloc_bench] trials=" << n << " accepted=" << conv << "\n";
    return 0;
}
