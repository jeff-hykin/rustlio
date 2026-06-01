#include "ivan_pgo.h"

#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <cmath>

using PointNT = pcl::PointNormal;
using CloudNT = pcl::PointCloud<PointNT>;

namespace
{
// Estimate normals on a PointXYZI cloud and return a PointNormal cloud
// (positions + normals) for point-to-plane ICP. Radius mirrors Ivan's
// estimate_normals(max_nn=30, radius=0.3).
CloudNT::Ptr with_normals(const CloudType::Ptr &cloud, double radius)
{
    CloudNT::Ptr out(new CloudNT);
    if (cloud->empty())
        return out;
    pcl::NormalEstimationOMP<PointType, pcl::Normal> ne;
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(radius);
    pcl::PointCloud<pcl::Normal> normals;
    ne.compute(normals);
    out->resize(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i)
    {
        out->points[i].x = cloud->points[i].x;
        out->points[i].y = cloud->points[i].y;
        out->points[i].z = cloud->points[i].z;
        out->points[i].normal_x = normals.points[i].normal_x;
        out->points[i].normal_y = normals.points[i].normal_y;
        out->points[i].normal_z = normals.points[i].normal_z;
    }
    out->width = out->size();
    out->height = 1;
    return out;
}
}  // namespace

IvanPGO::IvanPGO(const Config &config) : m_config(config)
{
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    m_isam2 = std::make_shared<gtsam::ISAM2>(params);
    m_initial_values.clear();
    m_graph.resize(0);
    m_r_offset.setIdentity();
    m_t_offset.setZero();
}

bool IvanPGO::isKeyPose(const PoseWithTime &pose)
{
    if (m_key_poses.empty())
        return true;
    const KeyPoseWithCloud &last = m_key_poses.back();
    double dt = (pose.t - last.t_local).norm();
    double dd = Eigen::Quaterniond(pose.r).angularDistance(Eigen::Quaterniond(last.r_local)) * 57.29578;
    return dt > m_config.key_pose_delta_trans || dd > m_config.key_pose_delta_deg;
}

bool IvanPGO::addKeyPose(const CloudWithPose &cwp)
{
    if (!isKeyPose(cwp.pose))
        return false;
    size_t idx = m_key_poses.size();
    M3D init_r = m_r_offset * cwp.pose.r;
    V3D init_t = m_r_offset * cwp.pose.t + m_t_offset;
    m_initial_values.insert(idx, gtsam::Pose3(gtsam::Rot3(init_r), gtsam::Point3(init_t)));
    if (idx == 0)
    {
        auto noise = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector6::Ones() * 1e-12);
        m_graph.add(gtsam::PriorFactor<gtsam::Pose3>(idx, gtsam::Pose3(gtsam::Rot3(init_r), gtsam::Point3(init_t)), noise));
    }
    else
    {
        const KeyPoseWithCloud &last = m_key_poses.back();
        M3D r_between = last.r_local.transpose() * cwp.pose.r;
        V3D t_between = last.r_local.transpose() * (cwp.pose.t - last.t_local);
        auto noise = gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-6).finished());
        m_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(idx - 1, idx, gtsam::Pose3(gtsam::Rot3(r_between), gtsam::Point3(t_between)), noise));
    }
    KeyPoseWithCloud item;
    item.time = cwp.pose.second;
    item.r_local = cwp.pose.r;
    item.t_local = cwp.pose.t;
    item.body_cloud = cwp.cloud;
    item.r_global = init_r;
    item.t_global = init_t;
    m_key_poses.push_back(item);
    return true;
}

CloudType::Ptr IvanPGO::getSubMap(int idx, int half_range, double resolution)
{
    int lo = std::max(0, idx - half_range);
    int hi = std::min(static_cast<int>(m_key_poses.size()) - 1, idx + half_range);
    CloudType::Ptr ret(new CloudType);
    for (int i = lo; i <= hi; ++i)
    {
        CloudType::Ptr g(new CloudType);
        pcl::transformPointCloud(*m_key_poses[i].body_cloud, *g, m_key_poses[i].t_global, Eigen::Quaterniond(m_key_poses[i].r_global));
        *ret += *g;
    }
    if (resolution > 0 && !ret->empty())
    {
        pcl::VoxelGrid<PointType> vg;
        vg.setLeafSize(resolution, resolution, resolution);
        vg.setInputCloud(ret);
        vg.filter(*ret);
    }
    return ret;
}

void IvanPGO::searchForLoopPairs()
{
    if (m_key_poses.size() < 10)
        return;
    double cur_time = m_key_poses.back().time;
    if (m_config.min_loop_detect_duration > 0.0 && m_last_loop_time >= 0.0 &&
        cur_time - m_last_loop_time < m_config.min_loop_detect_duration)
        return;

    size_t cur_idx = m_key_poses.size() - 1;
    const KeyPoseWithCloud &cur = m_key_poses.back();
    pcl::PointXYZ cur_pt;
    cur_pt.x = cur.t_global(0);
    cur_pt.y = cur.t_global(1);
    cur_pt.z = cur.t_global(2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr poses_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < m_key_poses.size() - 1; ++i)
        poses_cloud->push_back({static_cast<float>(m_key_poses[i].t_global(0)),
                                static_cast<float>(m_key_poses[i].t_global(1)),
                                static_cast<float>(m_key_poses[i].t_global(2))});
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(poses_cloud);
    std::vector<int> ids;
    std::vector<float> sqdists;
    if (kdtree.radiusSearch(cur_pt, m_config.loop_search_radius, ids, sqdists) == 0)
        return;

    // radiusSearch returns neighbours sorted by distance; take the NEAREST that
    // also satisfies the min time gap (Ivan sorts candidates and picks closest).
    int loop_idx = -1;
    for (size_t k = 0; k < ids.size(); ++k)
    {
        int i = ids[k];
        if (std::abs(cur.time - m_key_poses[i].time) > m_config.loop_time_tresh)
        {
            loop_idx = i;
            break;
        }
    }
    if (loop_idx == -1)
        return;

    CloudType::Ptr target = getSubMap(loop_idx, m_config.loop_submap_half_range, m_config.submap_resolution);
    CloudType::Ptr source = getSubMap(cur_idx, m_config.loop_source_submap_half_range, m_config.submap_resolution);
    if (const char *pfx = std::getenv("DUMP_CPP")) {
        static bool done = false;
        if (!done) {
            done = true;
            auto w = [](const std::string &p, const CloudType::Ptr &c) {
                std::ofstream f(p);
                for (auto &pt : c->points) f << pt.x << " " << pt.y << " " << pt.z << "\n";
            };
            w(std::string(pfx) + "_src.xyz", source);
            w(std::string(pfx) + "_tgt.xyz", target);
            std::cerr << "[ivan] dumped src=" << source->size() << " tgt=" << target->size()
                      << " (src_kf=" << cur_idx << " tgt_kf=" << loop_idx << ")\n";
        }
    }
    if (source->size() < static_cast<size_t>(std::max(10, 0)) || target->empty())
        return;

    // Point-to-plane ICP (normals on both clouds). Normals on the target give
    // ICP the plane constraints that stop it sliding along walls -- the core fix.
    double normal_radius = std::max(m_config.submap_resolution * 3.0, 0.3);
    CloudNT::Ptr src_n = with_normals(source, normal_radius);
    CloudNT::Ptr tgt_n = with_normals(target, normal_radius);

    pcl::IterativeClosestPointWithNormals<PointNT, PointNT> icp;
    icp.setMaximumIterations(50);
    icp.setMaxCorrespondenceDistance(m_config.max_icp_correspondence_dist);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setInputSource(src_n);
    icp.setInputTarget(tgt_n);
    CloudNT aligned;
    icp.align(aligned);

    if (!icp.hasConverged())
        return;
    double fitness = icp.getFitnessScore();  // mean squared corr. distance (m^2)
    if (fitness > m_config.loop_score_tresh)
        return;

    M4F T = icp.getFinalTransformation();
    M3D dR = T.block<3, 3>(0, 0).cast<double>();
    V3D dt = T.block<3, 1>(0, 3).cast<double>();
    M3D r_refined = dR * cur.r_global;
    V3D t_refined = dR * cur.t_global + dt;

    LoopPair pair;
    pair.source_id = cur_idx;
    pair.target_id = loop_idx;
    pair.score = fitness;
    pair.r_offset = m_key_poses[loop_idx].r_global.transpose() * r_refined;
    pair.t_offset = m_key_poses[loop_idx].r_global.transpose() * (t_refined - m_key_poses[loop_idx].t_global);

    if (m_config.max_loop_offset > 0.0 && pair.t_offset.norm() > m_config.max_loop_offset)
        return;

    m_cache_pairs.push_back(pair);
    m_history_pairs.emplace_back(pair.target_id, pair.source_id);
    m_last_loop_time = cur_time;
    if (std::getenv("ICP_LOG"))
        std::cerr << "[ivan loop] src=" << cur_idx << " tgt=" << loop_idx
                  << " fitness=" << fitness << " icp_t=" << dt.norm()
                  << " offset_t=" << pair.t_offset.norm() << "\n";
}

void IvanPGO::smoothAndUpdate()
{
    bool has_loop = !m_cache_pairs.empty();
    for (const LoopPair &pair : m_cache_pairs)
    {
        // Decoupled noise: translation variance = ICP fitness (m^2, floored at
        // 0.01 = 10 cm sigma); rotation variance a generous fixed 0.05 rad^2
        // (~13 deg). A uniform variance would conflate rad^2 and m^2 units.
        double trans_var = std::max(0.01, pair.score);
        double rot_var = 0.05;
        auto noise = gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << rot_var, rot_var, rot_var, trans_var, trans_var, trans_var).finished());
        m_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(pair.target_id, pair.source_id,
                                                       gtsam::Pose3(gtsam::Rot3(pair.r_offset), gtsam::Point3(pair.t_offset)), noise));
    }
    m_cache_pairs.clear();

    m_isam2->update(m_graph, m_initial_values);
    m_isam2->update();
    if (has_loop)
        for (int i = 0; i < 4; ++i)
            m_isam2->update();
    m_graph.resize(0);
    m_initial_values.clear();

    gtsam::Values est = m_isam2->calculateBestEstimate();
    for (size_t i = 0; i < m_key_poses.size(); ++i)
    {
        gtsam::Pose3 p = est.at<gtsam::Pose3>(i);
        m_key_poses[i].r_global = p.rotation().matrix();
        m_key_poses[i].t_global = p.translation();
    }
    const KeyPoseWithCloud &last = m_key_poses.back();
    m_r_offset = last.r_global * last.r_local.transpose();
    m_t_offset = last.t_global - m_r_offset * last.t_local;
}
