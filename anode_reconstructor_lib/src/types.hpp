#pragma once

#include <optional>

#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>

#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>



using Point = pcl::PointXYZRGB;
using Cloud = pcl::PointCloud<Point>;
using CloudPtr = Cloud::Ptr;
using CloudConstPtr = Cloud::ConstPtr;

using PointWithNormal = pcl::PointXYZRGBNormal;
using CloudWithNormals = pcl::PointCloud<PointWithNormal>;
using CloudWithNormalsPtr = CloudWithNormals::Ptr;
using CloudWithNormalsConstPtr = CloudWithNormals::ConstPtr;

using Open3dCloud = open3d::geometry::PointCloud;
using Open3dCloudPtr = std::shared_ptr<Open3dCloud>;
using Open3dMesh = open3d::geometry::TriangleMesh;
using Open3dMeshPtr = std::shared_ptr<Open3dMesh>;
using Open3dLineSet = open3d::geometry::LineSet;
using Open3dLineSetPtr = std::shared_ptr<Open3dLineSet>;
using Open3dAABB = open3d::geometry::AxisAlignedBoundingBox;


struct AABoundingBox {
	Eigen::Vector3d min;
	Eigen::Vector3d max;
};

struct AnodeBlockWall {
	CloudWithNormalsPtr cloud;
	Eigen::Vector3d normal;
	double coeff_d;
	Eigen::Vector3d bb_min;
	Eigen::Vector3d bb_max;
};

struct VisCtx {
	Open3dMeshPtr mesh_without_defects;
	std::optional<Open3dMeshPtr> mesh_with_defects;
	std::optional<Open3dLineSetPtr> wall_a_rect;
	std::optional<Open3dLineSetPtr> wall_b_rect;
	std::optional<Open3dCloudPtr> reconstructed_block;
	// bool show_coordinate_axis = false;
};

struct DefectsSearchResult {
	Open3dLineSetPtr lineset;
	double average_distance_to_wall;
	double mse;
	double rmse;
	double percent_of_damaged_points;
	int number_of_damaged_points;
	int number_of_total_points;
};
