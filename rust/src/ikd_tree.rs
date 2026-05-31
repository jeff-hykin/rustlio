//! Incremental KD-Tree (ikd-Tree) for robotic applications.
//!
//! Rust translation of the C++ ikd-Tree by Yixi Cai.
//! Single-threaded: all rebuilds happen inline (no background thread).

use crate::commons::Point;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSS: f32 = 1e-6;
const MINIMAL_UNBALANCED_TREE_SIZE: i32 = 10;
const DOWNSAMPLE_SWITCH: bool = true;

// ---------------------------------------------------------------------------
// BoxPointType
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct BoxPointType {
    pub vertex_min: [f32; 3],
    pub vertex_max: [f32; 3],
}

// ---------------------------------------------------------------------------
// Internal enums
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeletePointStorage {
    NotRecord,
    DeletePointsRec,
}

// ---------------------------------------------------------------------------
// Max-heap entry for nearest-neighbor search
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct PointDist {
    point: Point,
    dist: f32,
}

impl PartialEq for PointDist {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for PointDist {}

impl PartialOrd for PointDist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// BinaryHeap is a max-heap by default. We want the *largest* distance on
/// top so we can pop it when we find something closer -- that matches the
/// C++ MANUAL_HEAP which is also a max-heap.
impl Ord for PointDist {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// KdTreeNode
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct KdTreeNode {
    point: Point,
    division_axis: i32,
    tree_size: i32,
    invalid_point_num: i32,
    down_del_num: i32,
    point_deleted: bool,
    tree_deleted: bool,
    point_downsample_deleted: bool,
    tree_downsample_deleted: bool,
    need_push_down_to_left: bool,
    need_push_down_to_right: bool,
    node_range_x: [f32; 2],
    node_range_y: [f32; 2],
    node_range_z: [f32; 2],
    radius_sq: f32,
    left: Option<Box<KdTreeNode>>,
    right: Option<Box<KdTreeNode>>,
}

impl KdTreeNode {
    fn new() -> Self {
        Self {
            point: Point::default(),
            division_axis: 0,
            tree_size: 1,
            invalid_point_num: 0,
            down_del_num: 0,
            point_deleted: false,
            tree_deleted: false,
            point_downsample_deleted: false,
            tree_downsample_deleted: false,
            need_push_down_to_left: false,
            need_push_down_to_right: false,
            node_range_x: [0.0, 0.0],
            node_range_y: [0.0, 0.0],
            node_range_z: [0.0, 0.0],
            radius_sq: 0.0,
            left: None,
            right: None,
        }
    }
}

// ---------------------------------------------------------------------------
// KdTree
// ---------------------------------------------------------------------------

pub struct KdTree {
    root: Option<Box<KdTreeNode>>,
    delete_criterion_param: f32,
    balance_criterion_param: f32,
    downsample_size: f32,
    points_deleted: Vec<Point>,
}

// ---------------------------------------------------------------------------
// Helper functions (free-standing)
// ---------------------------------------------------------------------------

fn point_axis(p: &Point, axis: i32) -> f32 {
    match axis {
        0 => p.x,
        1 => p.y,
        _ => p.z,
    }
}

fn same_point(a: &Point, b: &Point) -> bool {
    (a.x - b.x).abs() < EPSS && (a.y - b.y).abs() < EPSS && (a.z - b.z).abs() < EPSS
}

fn calc_dist(a: &Point, b: &Point) -> f32 {
    (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)
}

fn calc_box_dist(node: &Option<Box<KdTreeNode>>, point: &Point) -> f32 {
    match node {
        None => f32::INFINITY,
        Some(n) => {
            let mut min_dist = 0.0f32;
            if point.x < n.node_range_x[0] {
                min_dist += (point.x - n.node_range_x[0]) * (point.x - n.node_range_x[0]);
            }
            if point.x > n.node_range_x[1] {
                min_dist += (point.x - n.node_range_x[1]) * (point.x - n.node_range_x[1]);
            }
            if point.y < n.node_range_y[0] {
                min_dist += (point.y - n.node_range_y[0]) * (point.y - n.node_range_y[0]);
            }
            if point.y > n.node_range_y[1] {
                min_dist += (point.y - n.node_range_y[1]) * (point.y - n.node_range_y[1]);
            }
            if point.z < n.node_range_z[0] {
                min_dist += (point.z - n.node_range_z[0]) * (point.z - n.node_range_z[0]);
            }
            if point.z > n.node_range_z[1] {
                min_dist += (point.z - n.node_range_z[1]) * (point.z - n.node_range_z[1]);
            }
            min_dist
        }
    }
}

/// Build a balanced sub-tree from `storage[l..=r]` (inclusive range).
fn build_tree(storage: &mut [Point], l: usize, r: usize) -> Option<Box<KdTreeNode>> {
    if l > r {
        return None;
    }
    let mut node = KdTreeNode::new();
    let mid = (l + r) / 2;

    // Find the axis with the largest range.
    let mut min_val = [f32::INFINITY; 3];
    let mut max_val = [f32::NEG_INFINITY; 3];
    for i in l..=r {
        let p = &storage[i];
        min_val[0] = min_val[0].min(p.x);
        min_val[1] = min_val[1].min(p.y);
        min_val[2] = min_val[2].min(p.z);
        max_val[0] = max_val[0].max(p.x);
        max_val[1] = max_val[1].max(p.y);
        max_val[2] = max_val[2].max(p.z);
    }
    let mut div_axis: i32 = 0;
    let mut max_range = max_val[0] - min_val[0];
    for ax in 1..3 {
        let range = max_val[ax] - min_val[ax];
        if range > max_range {
            max_range = range;
            div_axis = ax as i32;
        }
    }
    node.division_axis = div_axis;

    // nth_element equivalent: partial sort so that storage[mid] is the
    // median along div_axis, with smaller on the left and larger on the right.
    let slice = &mut storage[l..=r];
    let nth = mid - l;
    slice.select_nth_unstable_by(nth, |a, b| {
        point_axis(a, div_axis)
            .partial_cmp(&point_axis(b, div_axis))
            .unwrap_or(Ordering::Equal)
    });

    node.point = storage[mid];

    // Recurse -- need to be careful with underflow when mid == 0 or mid == l
    node.left = if mid > l {
        build_tree(storage, l, mid - 1)
    } else {
        // l == mid, no left children
        None
    };
    node.right = if mid < r {
        build_tree(storage, mid + 1, r)
    } else {
        None
    };

    update_node(&mut node);
    Some(Box::new(node))
}

// ---------------------------------------------------------------------------
// Update: recompute aggregated stats from children
// ---------------------------------------------------------------------------

fn update_node(root: &mut KdTreeNode) {
    let mut tmp_range_x = [f32::INFINITY, f32::NEG_INFINITY];
    let mut tmp_range_y = [f32::INFINITY, f32::NEG_INFINITY];
    let mut tmp_range_z = [f32::INFINITY, f32::NEG_INFINITY];

    match (&root.left, &root.right) {
        (Some(left), Some(right)) => {
            root.tree_size = left.tree_size + right.tree_size + 1;
            root.invalid_point_num = left.invalid_point_num
                + right.invalid_point_num
                + if root.point_deleted { 1 } else { 0 };
            root.down_del_num = left.down_del_num
                + right.down_del_num
                + if root.point_downsample_deleted { 1 } else { 0 };
            root.tree_downsample_deleted =
                left.tree_downsample_deleted & right.tree_downsample_deleted & root.point_downsample_deleted;
            root.tree_deleted =
                left.tree_deleted && right.tree_deleted && root.point_deleted;

            if root.tree_deleted
                || (!left.tree_deleted && !right.tree_deleted && !root.point_deleted)
            {
                tmp_range_x[0] = left.node_range_x[0].min(right.node_range_x[0]).min(root.point.x);
                tmp_range_x[1] = left.node_range_x[1].max(right.node_range_x[1]).max(root.point.x);
                tmp_range_y[0] = left.node_range_y[0].min(right.node_range_y[0]).min(root.point.y);
                tmp_range_y[1] = left.node_range_y[1].max(right.node_range_y[1]).max(root.point.y);
                tmp_range_z[0] = left.node_range_z[0].min(right.node_range_z[0]).min(root.point.z);
                tmp_range_z[1] = left.node_range_z[1].max(right.node_range_z[1]).max(root.point.z);
            } else {
                if !left.tree_deleted {
                    tmp_range_x[0] = tmp_range_x[0].min(left.node_range_x[0]);
                    tmp_range_x[1] = tmp_range_x[1].max(left.node_range_x[1]);
                    tmp_range_y[0] = tmp_range_y[0].min(left.node_range_y[0]);
                    tmp_range_y[1] = tmp_range_y[1].max(left.node_range_y[1]);
                    tmp_range_z[0] = tmp_range_z[0].min(left.node_range_z[0]);
                    tmp_range_z[1] = tmp_range_z[1].max(left.node_range_z[1]);
                }
                if !right.tree_deleted {
                    tmp_range_x[0] = tmp_range_x[0].min(right.node_range_x[0]);
                    tmp_range_x[1] = tmp_range_x[1].max(right.node_range_x[1]);
                    tmp_range_y[0] = tmp_range_y[0].min(right.node_range_y[0]);
                    tmp_range_y[1] = tmp_range_y[1].max(right.node_range_y[1]);
                    tmp_range_z[0] = tmp_range_z[0].min(right.node_range_z[0]);
                    tmp_range_z[1] = tmp_range_z[1].max(right.node_range_z[1]);
                }
                if !root.point_deleted {
                    tmp_range_x[0] = tmp_range_x[0].min(root.point.x);
                    tmp_range_x[1] = tmp_range_x[1].max(root.point.x);
                    tmp_range_y[0] = tmp_range_y[0].min(root.point.y);
                    tmp_range_y[1] = tmp_range_y[1].max(root.point.y);
                    tmp_range_z[0] = tmp_range_z[0].min(root.point.z);
                    tmp_range_z[1] = tmp_range_z[1].max(root.point.z);
                }
            }
        }
        (Some(child), None) | (None, Some(child)) => {
            root.tree_size = child.tree_size + 1;
            root.invalid_point_num =
                child.invalid_point_num + if root.point_deleted { 1 } else { 0 };
            root.down_del_num =
                child.down_del_num + if root.point_downsample_deleted { 1 } else { 0 };
            root.tree_downsample_deleted =
                child.tree_downsample_deleted & root.point_downsample_deleted;
            root.tree_deleted = child.tree_deleted && root.point_deleted;

            if root.tree_deleted || (!child.tree_deleted && !root.point_deleted) {
                tmp_range_x[0] = child.node_range_x[0].min(root.point.x);
                tmp_range_x[1] = child.node_range_x[1].max(root.point.x);
                tmp_range_y[0] = child.node_range_y[0].min(root.point.y);
                tmp_range_y[1] = child.node_range_y[1].max(root.point.y);
                tmp_range_z[0] = child.node_range_z[0].min(root.point.z);
                tmp_range_z[1] = child.node_range_z[1].max(root.point.z);
            } else {
                if !child.tree_deleted {
                    tmp_range_x[0] = tmp_range_x[0].min(child.node_range_x[0]);
                    tmp_range_x[1] = tmp_range_x[1].max(child.node_range_x[1]);
                    tmp_range_y[0] = tmp_range_y[0].min(child.node_range_y[0]);
                    tmp_range_y[1] = tmp_range_y[1].max(child.node_range_y[1]);
                    tmp_range_z[0] = tmp_range_z[0].min(child.node_range_z[0]);
                    tmp_range_z[1] = tmp_range_z[1].max(child.node_range_z[1]);
                }
                if !root.point_deleted {
                    tmp_range_x[0] = tmp_range_x[0].min(root.point.x);
                    tmp_range_x[1] = tmp_range_x[1].max(root.point.x);
                    tmp_range_y[0] = tmp_range_y[0].min(root.point.y);
                    tmp_range_y[1] = tmp_range_y[1].max(root.point.y);
                    tmp_range_z[0] = tmp_range_z[0].min(root.point.z);
                    tmp_range_z[1] = tmp_range_z[1].max(root.point.z);
                }
            }
        }
        (None, None) => {
            root.tree_size = 1;
            root.invalid_point_num = if root.point_deleted { 1 } else { 0 };
            root.down_del_num = if root.point_downsample_deleted { 1 } else { 0 };
            root.tree_downsample_deleted = root.point_downsample_deleted;
            root.tree_deleted = root.point_deleted;
            tmp_range_x = [root.point.x, root.point.x];
            tmp_range_y = [root.point.y, root.point.y];
            tmp_range_z = [root.point.z, root.point.z];
        }
    }

    root.node_range_x = tmp_range_x;
    root.node_range_y = tmp_range_y;
    root.node_range_z = tmp_range_z;
    let x_l = (root.node_range_x[1] - root.node_range_x[0]) * 0.5;
    let y_l = (root.node_range_y[1] - root.node_range_y[0]) * 0.5;
    let z_l = (root.node_range_z[1] - root.node_range_z[0]) * 0.5;
    root.radius_sq = x_l * x_l + y_l * y_l + z_l * z_l;
}

// ---------------------------------------------------------------------------
// Push-down lazy deletion flags to children
// ---------------------------------------------------------------------------

fn push_down(root: &mut KdTreeNode) {
    if root.need_push_down_to_left {
        if let Some(ref mut child) = root.left {
            child.tree_downsample_deleted |= root.tree_downsample_deleted;
            child.point_downsample_deleted |= root.tree_downsample_deleted;
            child.tree_deleted =
                root.tree_deleted || child.tree_downsample_deleted;
            child.point_deleted =
                child.tree_deleted || child.point_downsample_deleted;
            if root.tree_downsample_deleted {
                child.down_del_num = child.tree_size;
            }
            if root.tree_deleted {
                child.invalid_point_num = child.tree_size;
            } else {
                child.invalid_point_num = child.down_del_num;
            }
            child.need_push_down_to_left = true;
            child.need_push_down_to_right = true;
        }
        root.need_push_down_to_left = false;
    }
    if root.need_push_down_to_right {
        if let Some(ref mut child) = root.right {
            child.tree_downsample_deleted |= root.tree_downsample_deleted;
            child.point_downsample_deleted |= root.tree_downsample_deleted;
            child.tree_deleted =
                root.tree_deleted || child.tree_downsample_deleted;
            child.point_deleted =
                child.tree_deleted || child.point_downsample_deleted;
            if root.tree_downsample_deleted {
                child.down_del_num = child.tree_size;
            }
            if root.tree_deleted {
                child.invalid_point_num = child.tree_size;
            } else {
                child.invalid_point_num = child.down_del_num;
            }
            child.need_push_down_to_left = true;
            child.need_push_down_to_right = true;
        }
        root.need_push_down_to_right = false;
    }
}

// ---------------------------------------------------------------------------
// Flatten: collect all non-deleted points from a subtree
// ---------------------------------------------------------------------------

fn flatten_rec(
    node: &mut KdTreeNode,
    storage: &mut Vec<Point>,
    storage_type: DeletePointStorage,
    deleted_buf: &mut Vec<Point>,
) {
    push_down(node);
    if !node.point_deleted {
        storage.push(node.point);
    }
    // Need to temporarily take children to recurse with mutable refs.
    if let Some(mut left) = node.left.take() {
        flatten_rec(&mut left, storage, storage_type, deleted_buf);
        node.left = Some(left);
    }
    if let Some(mut right) = node.right.take() {
        flatten_rec(&mut right, storage, storage_type, deleted_buf);
        node.right = Some(right);
    }
    if storage_type == DeletePointStorage::DeletePointsRec {
        if node.point_deleted && !node.point_downsample_deleted {
            deleted_buf.push(node.point);
        }
    }
}

// ---------------------------------------------------------------------------
// Criterion check: should this subtree be rebuilt?
// ---------------------------------------------------------------------------

fn criterion_check(
    root: &KdTreeNode,
    delete_criterion_param: f32,
    balance_criterion_param: f32,
) -> bool {
    if root.tree_size <= MINIMAL_UNBALANCED_TREE_SIZE {
        return false;
    }
    let delete_evaluation = root.invalid_point_num as f32 / root.tree_size as f32;
    if delete_evaluation > delete_criterion_param {
        return true;
    }
    // Balance check: pick whichever child exists (prefer left).
    let son_size = root
        .left
        .as_ref()
        .map(|n| n.tree_size)
        .or_else(|| root.right.as_ref().map(|n| n.tree_size));
    if let Some(ss) = son_size {
        let balance_evaluation = ss as f32 / (root.tree_size - 1) as f32;
        if balance_evaluation > balance_criterion_param
            || balance_evaluation < 1.0 - balance_criterion_param
        {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Rebuild: flatten then rebuild balanced
// ---------------------------------------------------------------------------

fn rebuild(
    root: &mut Option<Box<KdTreeNode>>,
    delete_criterion_param: f32,
    balance_criterion_param: f32,
    points_deleted: &mut Vec<Point>,
) {
    if let Some(ref mut node) = root {
        let mut pcl_storage = Vec::new();
        flatten_rec(node, &mut pcl_storage, DeletePointStorage::DeletePointsRec, points_deleted);
        if pcl_storage.is_empty() {
            *root = None;
        } else {
            let len = pcl_storage.len();
            *root = build_tree(&mut pcl_storage, 0, len - 1);
        }
    }
    // After rebuild, no rebalancing check needed -- it's balanced by construction.
    let _ = (delete_criterion_param, balance_criterion_param);
}

// ---------------------------------------------------------------------------
// Recursive search
// ---------------------------------------------------------------------------

#[inline]
fn heap_top_dist(heap: &BinaryHeap<PointDist>) -> f32 {
    heap.peek().map_or(f32::INFINITY, |e| e.dist)
}

fn search_rec(
    node: &KdTreeNode,
    k: usize,
    point: &Point,
    heap: &mut BinaryHeap<PointDist>,
    max_dist_sqr: f32,
) {
    if node.tree_deleted {
        return;
    }
    let cur_dist = calc_box_dist_node(node, point);
    if cur_dist > max_dist_sqr {
        return;
    }

    // Note: We skip push_down here because search is &self (immutable).
    // The push_down state is only relevant for correctness of point_deleted
    // which is already propagated eagerly in the mutable operations.
    // For search we read the existing flags as-is.

    if !node.point_deleted {
        let dist = calc_dist(point, &node.point);
        if dist <= max_dist_sqr && (heap.len() < k || dist < heap.peek().map_or(f32::INFINITY, |e| e.dist)) {
            if heap.len() >= k {
                heap.pop();
            }
            heap.push(PointDist {
                point: node.point,
                dist,
            });
        }
    }

    let dist_left = calc_box_dist(&node.left, point);
    let dist_right = calc_box_dist(&node.right, point);

    let td = heap_top_dist(heap);

    if heap.len() < k || (dist_left < td && dist_right < td) {
        // Search both, nearer first
        if dist_left <= dist_right {
            if let Some(ref left) = node.left {
                search_rec(left, k, point, heap, max_dist_sqr);
            }
            if heap.len() < k || dist_right < heap_top_dist(heap) {
                if let Some(ref right) = node.right {
                    search_rec(right, k, point, heap, max_dist_sqr);
                }
            }
        } else {
            if let Some(ref right) = node.right {
                search_rec(right, k, point, heap, max_dist_sqr);
            }
            if heap.len() < k || dist_left < heap_top_dist(heap) {
                if let Some(ref left) = node.left {
                    search_rec(left, k, point, heap, max_dist_sqr);
                }
            }
        }
    } else {
        if dist_left < td {
            if let Some(ref left) = node.left {
                search_rec(left, k, point, heap, max_dist_sqr);
            }
        }
        if dist_right < heap_top_dist(heap) {
            if let Some(ref right) = node.right {
                search_rec(right, k, point, heap, max_dist_sqr);
            }
        }
    }
}

fn calc_box_dist_node(node: &KdTreeNode, point: &Point) -> f32 {
    let mut min_dist = 0.0f32;
    if point.x < node.node_range_x[0] {
        min_dist += (point.x - node.node_range_x[0]) * (point.x - node.node_range_x[0]);
    }
    if point.x > node.node_range_x[1] {
        min_dist += (point.x - node.node_range_x[1]) * (point.x - node.node_range_x[1]);
    }
    if point.y < node.node_range_y[0] {
        min_dist += (point.y - node.node_range_y[0]) * (point.y - node.node_range_y[0]);
    }
    if point.y > node.node_range_y[1] {
        min_dist += (point.y - node.node_range_y[1]) * (point.y - node.node_range_y[1]);
    }
    if point.z < node.node_range_z[0] {
        min_dist += (point.z - node.node_range_z[0]) * (point.z - node.node_range_z[0]);
    }
    if point.z > node.node_range_z[1] {
        min_dist += (point.z - node.node_range_z[1]) * (point.z - node.node_range_z[1]);
    }
    min_dist
}

// ---------------------------------------------------------------------------
// Search by range (box search)
// ---------------------------------------------------------------------------

fn search_by_range_rec(node: &KdTreeNode, bbox: &BoxPointType, storage: &mut Vec<Point>) {
    // Early-out: node bounding box fully outside query box
    if bbox.vertex_max[0] <= node.node_range_x[0] || bbox.vertex_min[0] > node.node_range_x[1] {
        return;
    }
    if bbox.vertex_max[1] <= node.node_range_y[0] || bbox.vertex_min[1] > node.node_range_y[1] {
        return;
    }
    if bbox.vertex_max[2] <= node.node_range_z[0] || bbox.vertex_min[2] > node.node_range_z[1] {
        return;
    }
    // Node bounding box fully inside query box: flatten all non-deleted points
    if bbox.vertex_min[0] <= node.node_range_x[0]
        && bbox.vertex_max[0] > node.node_range_x[1]
        && bbox.vertex_min[1] <= node.node_range_y[0]
        && bbox.vertex_max[1] > node.node_range_y[1]
        && bbox.vertex_min[2] <= node.node_range_z[0]
        && bbox.vertex_max[2] > node.node_range_z[1]
    {
        flatten_immutable(node, storage);
        return;
    }
    // Check the node's own point
    if !node.point_deleted
        && bbox.vertex_min[0] <= node.point.x
        && bbox.vertex_max[0] > node.point.x
        && bbox.vertex_min[1] <= node.point.y
        && bbox.vertex_max[1] > node.point.y
        && bbox.vertex_min[2] <= node.point.z
        && bbox.vertex_max[2] > node.point.z
    {
        storage.push(node.point);
    }
    if let Some(ref left) = node.left {
        search_by_range_rec(left, bbox, storage);
    }
    if let Some(ref right) = node.right {
        search_by_range_rec(right, bbox, storage);
    }
}

/// Immutable flatten -- collect all non-deleted points.
fn flatten_immutable(node: &KdTreeNode, storage: &mut Vec<Point>) {
    if node.tree_deleted {
        return;
    }
    if !node.point_deleted {
        storage.push(node.point);
    }
    if let Some(ref left) = node.left {
        flatten_immutable(left, storage);
    }
    if let Some(ref right) = node.right {
        flatten_immutable(right, storage);
    }
}

// ---------------------------------------------------------------------------
// Delete by range (mutable)
// ---------------------------------------------------------------------------

fn delete_by_range(
    root: &mut Option<Box<KdTreeNode>>,
    boxpoint: &BoxPointType,
    allow_rebuild: bool,
    is_downsample: bool,
    delete_criterion_param: f32,
    balance_criterion_param: f32,
    points_deleted: &mut Vec<Point>,
) -> i32 {
    let node = match root.as_mut() {
        Some(n) => n,
        None => return 0,
    };
    if node.tree_deleted {
        return 0;
    }
    push_down(node);

    // Early out: query box doesn't intersect node range
    if boxpoint.vertex_max[0] <= node.node_range_x[0]
        || boxpoint.vertex_min[0] > node.node_range_x[1]
    {
        return 0;
    }
    if boxpoint.vertex_max[1] <= node.node_range_y[0]
        || boxpoint.vertex_min[1] > node.node_range_y[1]
    {
        return 0;
    }
    if boxpoint.vertex_max[2] <= node.node_range_z[0]
        || boxpoint.vertex_min[2] > node.node_range_z[1]
    {
        return 0;
    }

    let mut tmp_counter: i32 = 0;

    // Query box fully covers node range: delete entire subtree
    if boxpoint.vertex_min[0] <= node.node_range_x[0]
        && boxpoint.vertex_max[0] > node.node_range_x[1]
        && boxpoint.vertex_min[1] <= node.node_range_y[0]
        && boxpoint.vertex_max[1] > node.node_range_y[1]
        && boxpoint.vertex_min[2] <= node.node_range_z[0]
        && boxpoint.vertex_max[2] > node.node_range_z[1]
    {
        node.tree_deleted = true;
        node.point_deleted = true;
        node.need_push_down_to_left = true;
        node.need_push_down_to_right = true;
        tmp_counter = node.tree_size - node.invalid_point_num;
        node.invalid_point_num = node.tree_size;
        if is_downsample {
            node.tree_downsample_deleted = true;
            node.point_downsample_deleted = true;
            node.down_del_num = node.tree_size;
        }
        return tmp_counter;
    }

    // Check node's own point
    if !node.point_deleted
        && boxpoint.vertex_min[0] <= node.point.x
        && boxpoint.vertex_max[0] > node.point.x
        && boxpoint.vertex_min[1] <= node.point.y
        && boxpoint.vertex_max[1] > node.point.y
        && boxpoint.vertex_min[2] <= node.point.z
        && boxpoint.vertex_max[2] > node.point.z
    {
        node.point_deleted = true;
        tmp_counter += 1;
        if is_downsample {
            node.point_downsample_deleted = true;
        }
    }

    // Recurse into children
    tmp_counter += delete_by_range(
        &mut node.left,
        boxpoint,
        allow_rebuild,
        is_downsample,
        delete_criterion_param,
        balance_criterion_param,
        points_deleted,
    );
    tmp_counter += delete_by_range(
        &mut node.right,
        boxpoint,
        allow_rebuild,
        is_downsample,
        delete_criterion_param,
        balance_criterion_param,
        points_deleted,
    );

    update_node(node);

    if allow_rebuild && criterion_check(node, delete_criterion_param, balance_criterion_param) {
        rebuild(root, delete_criterion_param, balance_criterion_param, points_deleted);
    }

    tmp_counter
}

// ---------------------------------------------------------------------------
// Delete by point (mutable)
// ---------------------------------------------------------------------------

fn delete_by_point(
    root: &mut Option<Box<KdTreeNode>>,
    point: &Point,
    allow_rebuild: bool,
    delete_criterion_param: f32,
    balance_criterion_param: f32,
    points_deleted: &mut Vec<Point>,
) {
    let node = match root.as_mut() {
        Some(n) => n,
        None => return,
    };
    if node.tree_deleted {
        return;
    }
    push_down(node);

    if same_point(&node.point, point) && !node.point_deleted {
        node.point_deleted = true;
        node.invalid_point_num += 1;
        if node.invalid_point_num == node.tree_size {
            node.tree_deleted = true;
        }
        return;
    }

    let go_left = (node.division_axis == 0 && point.x < node.point.x)
        || (node.division_axis == 1 && point.y < node.point.y)
        || (node.division_axis == 2 && point.z < node.point.z);

    if go_left {
        delete_by_point(
            &mut node.left,
            point,
            allow_rebuild,
            delete_criterion_param,
            balance_criterion_param,
            points_deleted,
        );
    } else {
        delete_by_point(
            &mut node.right,
            point,
            allow_rebuild,
            delete_criterion_param,
            balance_criterion_param,
            points_deleted,
        );
    }

    update_node(node);

    if allow_rebuild && criterion_check(node, delete_criterion_param, balance_criterion_param) {
        rebuild(root, delete_criterion_param, balance_criterion_param, points_deleted);
    }
}

// ---------------------------------------------------------------------------
// Add by point (mutable)
// ---------------------------------------------------------------------------

fn add_by_point(
    root: &mut Option<Box<KdTreeNode>>,
    point: Point,
    allow_rebuild: bool,
    father_axis: i32,
    delete_criterion_param: f32,
    balance_criterion_param: f32,
    points_deleted: &mut Vec<Point>,
) {
    if root.is_none() {
        let mut node = KdTreeNode::new();
        node.point = point;
        node.division_axis = (father_axis + 1) % 3;
        update_node(&mut node);
        *root = Some(Box::new(node));
        return;
    }

    let div_axis;
    {
        let node = root.as_mut().unwrap();
        push_down(node);
        div_axis = node.division_axis;

        let go_left = (div_axis == 0 && point.x < node.point.x)
            || (div_axis == 1 && point.y < node.point.y)
            || (div_axis == 2 && point.z < node.point.z);

        if go_left {
            add_by_point(
                &mut node.left,
                point,
                allow_rebuild,
                div_axis,
                delete_criterion_param,
                balance_criterion_param,
                points_deleted,
            );
        } else {
            add_by_point(
                &mut node.right,
                point,
                allow_rebuild,
                div_axis,
                delete_criterion_param,
                balance_criterion_param,
                points_deleted,
            );
        }
        update_node(node);
    }

    if allow_rebuild {
        let needs_rebuild = {
            let node = root.as_ref().unwrap();
            criterion_check(node, delete_criterion_param, balance_criterion_param)
        };
        if needs_rebuild {
            rebuild(root, delete_criterion_param, balance_criterion_param, points_deleted);
        }
    }
}

// ---------------------------------------------------------------------------
// Search by range for downsample (mutable -- needs push_down)
// ---------------------------------------------------------------------------

fn search_by_range_mut(
    node: &mut KdTreeNode,
    bbox: &BoxPointType,
    storage: &mut Vec<Point>,
) {
    push_down(node);

    if bbox.vertex_max[0] <= node.node_range_x[0] || bbox.vertex_min[0] > node.node_range_x[1] {
        return;
    }
    if bbox.vertex_max[1] <= node.node_range_y[0] || bbox.vertex_min[1] > node.node_range_y[1] {
        return;
    }
    if bbox.vertex_max[2] <= node.node_range_z[0] || bbox.vertex_min[2] > node.node_range_z[1] {
        return;
    }

    // Fully contained
    if bbox.vertex_min[0] <= node.node_range_x[0]
        && bbox.vertex_max[0] > node.node_range_x[1]
        && bbox.vertex_min[1] <= node.node_range_y[0]
        && bbox.vertex_max[1] > node.node_range_y[1]
        && bbox.vertex_min[2] <= node.node_range_z[0]
        && bbox.vertex_max[2] > node.node_range_z[1]
    {
        // Flatten all non-deleted
        let mut dummy = Vec::new();
        flatten_rec(node, storage, DeletePointStorage::NotRecord, &mut dummy);
        return;
    }

    if !node.point_deleted
        && bbox.vertex_min[0] <= node.point.x
        && bbox.vertex_max[0] > node.point.x
        && bbox.vertex_min[1] <= node.point.y
        && bbox.vertex_max[1] > node.point.y
        && bbox.vertex_min[2] <= node.point.z
        && bbox.vertex_max[2] > node.point.z
    {
        storage.push(node.point);
    }

    if let Some(ref mut left) = node.left {
        search_by_range_mut(left, bbox, storage);
    }
    if let Some(ref mut right) = node.right {
        search_by_range_mut(right, bbox, storage);
    }
}

// ---------------------------------------------------------------------------
// KdTree impl
// ---------------------------------------------------------------------------

impl KdTree {
    pub fn new(delete_param: f32, balance_param: f32, box_length: f32) -> Self {
        Self {
            root: None,
            delete_criterion_param: delete_param,
            balance_criterion_param: balance_param,
            downsample_size: box_length,
            points_deleted: Vec::new(),
        }
    }

    pub fn set_downsample_param(&mut self, downsample_param: f32) {
        self.downsample_size = downsample_param;
    }

    pub fn build(&mut self, mut points: Vec<Point>) {
        self.root = None;
        if points.is_empty() {
            return;
        }
        let len = points.len();
        self.root = build_tree(&mut points, 0, len - 1);
    }

    pub fn nearest_search(
        &self,
        point: &Point,
        k: usize,
        max_dist: f32,
    ) -> (Vec<Point>, Vec<f32>) {
        let mut heap = BinaryHeap::with_capacity(2 * k);
        let max_dist_sqr = max_dist * max_dist;

        if let Some(ref node) = self.root {
            search_rec(node, k, point, &mut heap, max_dist_sqr);
        }

        let k_found = k.min(heap.len());
        let mut points = Vec::with_capacity(k_found);
        let mut dists = Vec::with_capacity(k_found);
        // Pop from max-heap gives largest first; reverse to get nearest-first
        // (matches C++ which inserts at begin)
        while let Some(entry) = heap.pop() {
            points.push(entry.point);
            dists.push(entry.dist);
        }
        points.reverse();
        dists.reverse();
        (points, dists)
    }

    pub fn add_points(&mut self, points: &[Point], downsample_on: bool) -> i32 {
        let downsample_switch = downsample_on && DOWNSAMPLE_SWITCH;
        let mut tmp_counter = 0i32;

        for &pt in points {
            if downsample_switch {
                let ds = self.downsample_size;
                let mut box_of_point = BoxPointType::default();
                box_of_point.vertex_min[0] = (pt.x / ds).floor() * ds;
                box_of_point.vertex_max[0] = box_of_point.vertex_min[0] + ds;
                box_of_point.vertex_min[1] = (pt.y / ds).floor() * ds;
                box_of_point.vertex_max[1] = box_of_point.vertex_min[1] + ds;
                box_of_point.vertex_min[2] = (pt.z / ds).floor() * ds;
                box_of_point.vertex_max[2] = box_of_point.vertex_min[2] + ds;

                let mid_point = Point {
                    x: box_of_point.vertex_min[0] + ds * 0.5,
                    y: box_of_point.vertex_min[1] + ds * 0.5,
                    z: box_of_point.vertex_min[2] + ds * 0.5,
                    ..Point::default()
                };

                // Find existing points in this voxel
                let mut downsample_storage = Vec::new();
                if let Some(ref mut node) = self.root {
                    search_by_range_mut(node, &box_of_point, &mut downsample_storage);
                }

                let mut min_dist = calc_dist(&pt, &mid_point);
                let mut downsample_result = pt;
                for existing in &downsample_storage {
                    let tmp_dist = calc_dist(existing, &mid_point);
                    if tmp_dist < min_dist {
                        min_dist = tmp_dist;
                        downsample_result = *existing;
                    }
                }

                if downsample_storage.len() > 1 || same_point(&pt, &downsample_result) {
                    if !downsample_storage.is_empty() {
                        delete_by_range(
                            &mut self.root,
                            &box_of_point,
                            true,
                            true,
                            self.delete_criterion_param,
                            self.balance_criterion_param,
                            &mut self.points_deleted,
                        );
                    }
                    let div_axis = self
                        .root
                        .as_ref()
                        .map(|n| n.division_axis)
                        .unwrap_or(0);
                    add_by_point(
                        &mut self.root,
                        downsample_result,
                        true,
                        div_axis,
                        self.delete_criterion_param,
                        self.balance_criterion_param,
                        &mut self.points_deleted,
                    );
                    tmp_counter += 1;
                }
            } else {
                let div_axis = self
                    .root
                    .as_ref()
                    .map(|n| n.division_axis)
                    .unwrap_or(0);
                add_by_point(
                    &mut self.root,
                    pt,
                    true,
                    div_axis,
                    self.delete_criterion_param,
                    self.balance_criterion_param,
                    &mut self.points_deleted,
                );
            }
        }
        tmp_counter
    }

    pub fn delete_point_boxes(&mut self, boxes: &[BoxPointType]) -> i32 {
        let mut tmp_counter = 0i32;
        for bbox in boxes {
            tmp_counter += delete_by_range(
                &mut self.root,
                bbox,
                true,
                false,
                self.delete_criterion_param,
                self.balance_criterion_param,
                &mut self.points_deleted,
            );
        }
        tmp_counter
    }

    pub fn acquire_removed_points(&mut self) -> Vec<Point> {
        let removed = std::mem::take(&mut self.points_deleted);
        removed
    }

    pub fn size(&self) -> i32 {
        self.root.as_ref().map(|n| n.tree_size).unwrap_or(0)
    }

    pub fn validnum(&self) -> i32 {
        self.root
            .as_ref()
            .map(|n| n.tree_size - n.invalid_point_num)
            .unwrap_or(0)
    }

    pub fn box_search(&self, bbox: &BoxPointType) -> Vec<Point> {
        let mut storage = Vec::new();
        if let Some(ref node) = self.root {
            search_by_range_rec(node, bbox, &mut storage);
        }
        storage
    }

    /// Collect all non-deleted points in the tree.
    pub fn flatten(&self) -> Vec<Point> {
        let mut storage = Vec::new();
        if let Some(ref node) = self.root {
            flatten_immutable(node, &mut storage);
        }
        storage
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(x: f32, y: f32, z: f32) -> Point {
        Point {
            x,
            y,
            z,
            intensity: 0.0,
            curvature: 0.0,
        }
    }

    /// Simple deterministic pseudo-random for test reproducibility.
    fn pseudo_random_points(n: usize, seed: u64) -> Vec<Point> {
        let mut state = seed;
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = ((state >> 33) as f32) / (u32::MAX as f32) * 200.0 - 100.0;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = ((state >> 33) as f32) / (u32::MAX as f32) * 200.0 - 100.0;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let z = ((state >> 33) as f32) / (u32::MAX as f32) * 200.0 - 100.0;
            pts.push(make_point(x, y, z));
        }
        pts
    }

    #[test]
    fn test_build_and_search() {
        let pts = pseudo_random_points(100, 42);
        let mut tree = KdTree::new(0.5, 0.6, 0.2);
        tree.build(pts.clone());

        assert_eq!(tree.size(), 100);
        assert_eq!(tree.validnum(), 100);

        // Search for 5 nearest neighbors of the first point
        let query = pts[0];
        let (found, dists) = tree.nearest_search(&query, 5, f32::INFINITY);
        assert_eq!(found.len(), 5);
        assert_eq!(dists.len(), 5);
        // The first result should be the query point itself (distance 0)
        assert!(dists[0] < EPSS);
        // Distances should be sorted ascending
        for i in 1..dists.len() {
            assert!(dists[i] >= dists[i - 1]);
        }

        // Search with very small max_dist should return fewer or no results
        let (_found2, dists2) = tree.nearest_search(&query, 5, 0.001);
        // At minimum the point itself should be found (dist=0 < 0.001^2)
        assert!(!dists2.is_empty());
    }

    #[test]
    fn test_add_and_delete() {
        let pts = pseudo_random_points(50, 123);
        let mut tree = KdTree::new(0.5, 0.6, 0.2);
        tree.build(pts);
        assert_eq!(tree.size(), 50);

        // Add 20 more points (no downsample)
        let new_pts = pseudo_random_points(20, 456);
        tree.add_points(&new_pts, false);
        assert_eq!(tree.size(), 70);
        assert_eq!(tree.validnum(), 70);

        // Delete points in a box region around origin
        let del_box = BoxPointType {
            vertex_min: [-10.0, -10.0, -10.0],
            vertex_max: [10.0, 10.0, 10.0],
        };
        let deleted_count = tree.delete_point_boxes(&[del_box]);
        assert!(deleted_count >= 0);
        // Size stays the same (lazy deletion), but validnum decreases
        assert_eq!(tree.size(), 70);
        assert_eq!(tree.validnum(), 70 - deleted_count);

        // Box search in deleted region should return nothing
        let found = tree.box_search(&del_box);
        assert!(found.is_empty());

        // Flatten should return only the valid points
        let all = tree.flatten();
        assert_eq!(all.len() as i32, tree.validnum());
    }

    #[test]
    fn test_downsample_add() {
        let mut tree = KdTree::new(0.5, 0.6, 1.0);
        // Build with a single point
        tree.build(vec![make_point(0.1, 0.1, 0.1)]);
        assert_eq!(tree.size(), 1);

        // Add a nearby point with downsample on -- it should replace, not grow
        tree.add_points(&[make_point(0.2, 0.2, 0.2)], true);
        // After downsample, the voxel [0,1)^3 should have exactly 1 point
        // (the one closer to voxel center 0.5, 0.5, 0.5)
        let valid = tree.validnum();
        assert_eq!(valid, 1);
    }
}
