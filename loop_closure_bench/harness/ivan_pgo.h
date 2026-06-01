#pragma once
// Ivan-style PGO loop closure, ported from dimos pgo.py (branch
// autoresearch/ivan/pgo_go2). Same GTSAM iSAM2 backbone as SimplePGO, but the
// loop-closure registration is the part that differs and matters:
//
//   * point-to-plane ICP (target normals) instead of point-to-point -- normals
//     resolve the "slide along a wall" ambiguity that made the stock point-to-
//     point ICP accept multi-metre false alignments;
//   * bounded max correspondence distance (default 1.0 m, not 10 m);
//   * nearest in-radius candidate (sorted), not the first one found;
//   * decoupled noise model -- translation variance = ICP fitness (m²),
//     rotation variance a fixed generous 0.05 rad² (~13°), since loops aren't
//     trusted to fix rotation tightly without richer features.
//
// Reuses Config + the shared types from simple_pgo.h so the benchmark harness
// can drive either implementation through the same interface.
#include "pgos/simple_pgo.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <memory>
#include <vector>

class IvanPGO
{
public:
    explicit IvanPGO(const Config &config);

    bool addKeyPose(const CloudWithPose &cloud_with_pose);  // returns true if a keyframe was added
    void searchForLoopPairs();
    void smoothAndUpdate();

    std::vector<KeyPoseWithCloud> &keyPoses() { return m_key_poses; }
    std::vector<LoopPair> &cachePairs() { return m_cache_pairs; }
    std::vector<std::pair<size_t, size_t>> &historyPairs() { return m_history_pairs; }

private:
    bool isKeyPose(const PoseWithTime &pose);
    CloudType::Ptr getSubMap(int idx, int half_range, double resolution);

    Config m_config;
    std::vector<KeyPoseWithCloud> m_key_poses;
    std::vector<LoopPair> m_cache_pairs;
    std::vector<std::pair<size_t, size_t>> m_history_pairs;
    double m_last_loop_time = -1.0;

    std::shared_ptr<gtsam::ISAM2> m_isam2;
    gtsam::NonlinearFactorGraph m_graph;
    gtsam::Values m_initial_values;
    M3D m_r_offset;
    V3D m_t_offset;
};
