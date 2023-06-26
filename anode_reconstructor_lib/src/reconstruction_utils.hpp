#pragma once

#include <pcl/common/angles.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree.h>
#include <pcl/pcl_base.h>
#include <pcl/point_cloud.h>
#include <pcl/range_image/range_image.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/write_ply_points.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Shape_detection/Efficient_RANSAC.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/property_map.h>

#include <open3d/geometry/PointCloud.h>

#include "types.hpp"


typedef CGAL::Exact_predicates_inexact_constructions_kernel CgalKernel;
typedef std::pair<CgalKernel::Point_3, CgalKernel::Vector_3> CgalPointWithNormal;
typedef CGAL::First_of_pair_property_map<CgalPointWithNormal> CgalPointMap;
typedef CGAL::Second_of_pair_property_map<CgalPointWithNormal> CgalNormalMap;
typedef CGAL::Shape_detection::
	Efficient_RANSAC_traits<CgalKernel, std::vector<CgalPointWithNormal>, CgalPointMap, CgalNormalMap>
		CgalTraits;
typedef CGAL::Shape_detection::Efficient_RANSAC<CgalTraits> CgalEfficientRansac;
typedef CGAL::Shape_detection::Plane<CgalTraits> CgalPlane;
typedef CGAL::Parallel_if_available_tag CgalConcurrencyTag;

static const auto& cgal_params = CGAL::parameters::point_map(CgalPointMap()).normal_map(CgalNormalMap());


void ReconstructNormals(std::vector<CgalPointWithNormal>& points_with_normals);
void AddMissingWalls(CloudWithNormalsPtr cloud,
					 const Eigen::Vector3d& wall_bb_min,
					 const Eigen::Vector3d& block_bb_min,
					 const Eigen::Vector3d& block_bb_max);
auto RunCgalRansac(std::vector<CgalPointWithNormal> points_with_normals)
	-> std::tuple<CloudWithNormalsPtr, Eigen::Vector3d, double>;
auto CgalToPcl(std::vector<CgalPointWithNormal>& points_with_normals) -> CloudWithNormalsPtr;


template <typename PointT>
auto PclToOpen3d(std::shared_ptr<pcl::PointCloud<PointT>> cloud) -> std::shared_ptr<open3d::geometry::PointCloud>
{
	Open3dCloudPtr pcd(new Open3dCloud);

	for (auto& p : cloud->points) {
		pcd->points_.emplace_back(p.x, p.y, p.z);

		if (std::is_same<PointT, PointWithNormal>::value) {
			pcd->normals_.emplace_back(p.normal_x, p.normal_y, p.normal_z);
		}
	}

	return pcd;
}

template <typename PointT>
inline auto FilterByBoundingBox(std::shared_ptr<pcl::PointCloud<PointT>> cloud,
								Eigen::Vector3d min_point,
								Eigen::Vector3d max_point) -> std::shared_ptr<pcl::PointCloud<PointT>>
{
	typename pcl::octree::OctreePointCloudSearch<PointT> octree(128);
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();
	std::vector<int> indices;
	octree.boxSearch(min_point.cast<float>(), max_point.cast<float>(), indices);
	auto filtered_cloud = std::make_shared<pcl::PointCloud<PointT>>();
	pcl::copyPointCloud(*cloud, indices, *filtered_cloud);
	return filtered_cloud;
}

template <typename PointT>
inline auto PclToCgal(typename std::shared_ptr<pcl::PointCloud<PointT>>& cloud) -> std::vector<CgalPointWithNormal>
{
	std::vector<CgalPointWithNormal> points_with_normals;

	for (auto& p : cloud->points) {
		auto point = CgalKernel::Point_3(p.x, p.y, p.z);
		auto normal = CgalKernel::Vector_3(0, 0, 0);
		if constexpr (std::is_same<PointT, pcl::PointXYZRGBNormal>::value)
			normal = CgalKernel::Vector_3(p.normal_x, p.normal_y, p.normal_z);
		points_with_normals.emplace_back(CgalPointWithNormal(point, normal));
	}

	return points_with_normals;
}

template <typename PointT>
void StatisticalOutlierRemoval(typename std::shared_ptr<pcl::PointCloud<PointT>> cloud)
{
	int oldCloudSize = cloud->size();
	typename pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);

	// Create the filtering object
	pcl::StatisticalOutlierRemoval<PointT> sor;
	sor.setInputCloud(cloud);
	sor.setMeanK(50);
	sor.setStddevMulThresh(0.5);
	sor.filter(*cloud_filtered);

	cloud->points = cloud_filtered->points;
	cloud->width = cloud_filtered->width;
	cloud->height = cloud_filtered->height;
	// println("Result of 'statisticalOutlierRemoval'. Point Count. Before: '{}', After: '{}'", cloud->size(), oldCloudSize);
}

template <typename PointT>
void RotatePointCloud(std::shared_ptr<pcl::PointCloud<PointT>> cloud, float rotateX, float rotateY, float rotateZ)
{
	float radX = pcl::deg2rad(rotateX), radY = pcl::deg2rad(rotateY), radZ = pcl::deg2rad(rotateZ);
	Eigen::Affine3f transform = Eigen::Affine3f::Identity();
	transform.rotate(Eigen::AngleAxisf(radX, Eigen::Vector3f::UnitX()));
	transform.rotate(Eigen::AngleAxisf(radY, Eigen::Vector3f::UnitY()));
	transform.rotate(Eigen::AngleAxisf(radZ, Eigen::Vector3f::UnitZ()));
	// printf ("\nRotation matrix:\n");
	// std::cout << transform.matrix() << std::endl;

	pcl::transformPointCloud(*cloud, *cloud, transform);
}

template <typename PointT>
inline AABoundingBox GetAABoundingBox(std::shared_ptr<pcl::PointCloud<PointT>> cloud)
{
	pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
	feature_extractor.setInputCloud(cloud);
	feature_extractor.compute();
	PointT min_point, max_point;
	feature_extractor.getAABB(min_point, max_point);

	Eigen::Vector3f min{min_point.data};
	Eigen::Vector3f max(max_point.data);
	return AABoundingBox{min.cast<double>(), max.cast<double>()};
}

template <typename... T>
void Println(fmt::format_string<T...> fmt, T&&... args)
{
	fmt::print(fmt, std::forward<T>(args)...);
	fmt::print("\n");
	fflush(stdout);
}
