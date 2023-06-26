#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <open3d/geometry/Geometry.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/visualization/visualizer/Visualizer.h>
#include <optional>
#include <ostream>
#include <pcl/common/io.h>
#include <tuple>
#include <vector>

#include <fmt/core.h>

#include <gmpxx.h>

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/traits.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/PCLPointCloud2.h>
#include <pcl/PolygonMesh.h>
#include <pcl/common/centroid.h>
#include <pcl/console/time.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transformation_estimation_2D.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/warp_point_rigid_3d.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/poisson.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <open3d/Open3D.h>
#include <open3d/geometry/BoundingVolume.h>
#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/visualization/utility/Draw.h>
#include <open3d/visualization/utility/DrawGeometry.h>
#include <open3d/visualization/visualizer/RenderOption.h>

#include "icp2d_custom_warp_point.hpp"
#include "load_point_cloud.hpp"
#include "reconstruction_utils.hpp"


void RunMeshVisualizer(const VisCtx& ctx);

double GetPointToPlaneDistance(const Eigen::Vector3d& point, Eigen::Vector3d& normal, double d);

CloudWithNormalsPtr RunIcp(const CloudWithNormalsPtr target, const CloudWithNormalsPtr source);

void AlignScans(CloudWithNormalsPtr base, CloudWithNormalsPtr sub_mut, float lidar_dist, bool use_icp);

void RemoveConveyorBeltGround(CloudPtr cloud, double lidar_height);

AnodeBlockWall ProcessHalfOfAnodeBlock(CloudPtr point_cloud, int num, std::string debug_path);

void GetWallAngleAndRotateAnodeBlock(CloudPtr cloud_a, CloudPtr cloud_b);

DefectsSearchResult FindDefectsInOneWall(int wall_n,
										 Open3dMeshPtr poisson_mesh,
										 Eigen::Vector3d orig_wall_bb_max,
										 Eigen::Vector3d orig_wall_bb_min,
										 AABoundingBox full_block_bb,
										 CloudWithNormalsPtr wall_cloud,
										 double wall_dist_threshold = 0.005);

void ApplyRegionGrowing(Open3dMeshPtr mesh);


template <typename PointT>
void RunCloudDownsampling(typename std::shared_ptr<pcl::PointCloud<PointT>> cloud, float leaf_size)
{
	pcl::PCLPointCloud2::Ptr cloud_pcl2(new pcl::PCLPointCloud2());
	pcl::toPCLPointCloud2(*cloud, *cloud_pcl2);

	pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
	sor.setInputCloud(cloud_pcl2);
	sor.setLeafSize(leaf_size, leaf_size, leaf_size);
	sor.filter(*cloud_pcl2);

	pcl::fromPCLPointCloud2(*cloud_pcl2, *cloud);
};
