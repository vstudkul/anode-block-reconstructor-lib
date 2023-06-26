#include <CGAL/Random.h>
#include <fmt/core.h>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/write_ply_points.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Shape_detection/Efficient_RANSAC.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/property_map.h>

#include <open3d/geometry/Geometry.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/visualization/utility/Draw.h>
#include <open3d/visualization/utility/DrawGeometry.h>
#include <open3d/visualization/visualizer/RenderOption.h>

#include "reconstruction_utils.hpp"


auto CgalToPcl(std::vector<CgalPointWithNormal>& points_with_normals) -> CloudWithNormalsPtr
{
	CloudWithNormalsPtr cloud(new CloudWithNormals);
	for (auto& p : points_with_normals) {
		cloud->emplace_back(p.first.x(), p.first.y(), p.first.z(), 0, 200, 0, p.second.x(), p.second.y(), p.second.z());
	}

	return cloud;
}

auto RunCgalRansac(std::vector<CgalPointWithNormal> points_with_normals)
	-> std::tuple<CloudWithNormalsPtr, Eigen::Vector3d, double>
{
	// println("Start 'cgal_ransac': Cloud: {}", points_with_normals.size());

	CgalEfficientRansac ransac;
	ransac.set_input(points_with_normals, CgalPointMap(), CgalNormalMap());
	ransac.add_shape_factory<CgalPlane>();

	CgalEfficientRansac::Parameters params;
	// Set probability to miss the largest primitive at each iteration.
	// params.probability = 0.02;
	// params.probability = 0.02;
	// Detect shapes with at least 200 points_with_normals.
	//	params.min_points = 1500;
	// Set maximum Euclidean distance between a point and a shape.
	params.epsilon = 0.01;
	// params.epsilon = params_epsilon;
	// params.epsilon = 0.005;
	// Set maximum Euclidean distance between points_with_normals to be clustered.
	//	params.cluster_epsilon = 0.1;
	// Set maximum normal deviation.
	// 0.9 < dot(surface_normal, point_normal);
	//	params.normal_threshold = 0.8;

	ransac.detect(params);
	// println("Planes found: {}", ransac.planes().size());

	const auto& ransac_planes = ransac.planes();
	std::vector<boost::shared_ptr<CgalPlane>> planes_vec(ransac_planes.begin(), ransac_planes.end());
	std::sort(planes_vec.begin(), planes_vec.end(), [](auto& a, auto& b) {
		return a->indices_of_assigned_points().size() > b->indices_of_assigned_points().size();
	});

	CloudWithNormalsPtr biggest_cluster_points(new CloudWithNormals);
	auto& biggest_plane = planes_vec.at(0);
	for (const auto& i : biggest_plane->indices_of_assigned_points()) {
		auto& p = points_with_normals[i];
		biggest_cluster_points->emplace_back(p.first.x(), p.first.y(), p.first.z(), 0, 200, 0, p.second.x(),
											 p.second.y(), p.second.z());
	}

	CgalKernel::Vector_3 normal = biggest_plane->plane_normal();
	double d = biggest_plane->d();

	return {biggest_cluster_points, Eigen::Vector3d{normal.x(), normal.y(), normal.z()}, d};
}

void AddMissingWalls(CloudWithNormalsPtr cloud,
					 const Eigen::Vector3d& wall_bb_max,
					 const Eigen::Vector3d& block_bb_min,
					 const Eigen::Vector3d& block_bb_max)
{
	// println("WallBBmax: {}; BlockBBmax: {}; BlockBBmin: {}", wall_bb_max, block_bb_max, block_bb_min);

	// Build floor and left/right walls
	float floor_z = block_bb_min.z();
	float left_wall_x = block_bb_min.x();
	float right_wall_x = block_bb_max.x();

	cloud->points.reserve(cloud->points.size() + 3 * 100 * 100);

	int start_i = 3;
	int end_i = 97;

	// build floor
	for (int y = start_i; y < end_i; y++) {
		float y_pos = (block_bb_max.y() - block_bb_min.y()) / 100.0f * y;
		for (int x = start_i; x < end_i; x++) {
			float x_pos = (block_bb_max.x() - block_bb_min.x()) / 100.0f * x;
			auto p =
				PointWithNormal(block_bb_min.x() + x_pos, block_bb_min.y() + y_pos, floor_z, 255, 255, 0, 0, 0, -1);
			cloud->points.push_back(p);
		}
	}

	// build left wall
	for (int y = start_i; y < end_i; y++) {
		float y_pos = (block_bb_max.y() - block_bb_min.y()) / 100.0f * y;
		for (int z = start_i; z < end_i; z++) {
			float z_pos = (wall_bb_max.z() - block_bb_min.z()) / 100.0f * z;
			auto p =
				PointWithNormal(left_wall_x, block_bb_min.y() + y_pos, block_bb_min.z() + z_pos, 255, 255, 0, -1, 0, 0);
			cloud->points.push_back(p);
		}
	}

	// build right wall
	for (int y = start_i; y < end_i; y++) {
		float y_pos = (block_bb_max.y() - block_bb_min.y()) / 100.0f * y;
		for (int z = start_i; z < end_i; z++) {
			float z_pos = (wall_bb_max.z() - block_bb_min.z()) / 100.0f * z;
			auto p =
				PointWithNormal(right_wall_x, block_bb_min.y() + y_pos, block_bb_min.z() + z_pos, 255, 255, 0, 1, 0, 0);
			cloud->points.push_back(p);
		}
	}

	cloud->width = cloud->size();
	cloud->height = 1;
}

void ReconstructNormals(std::vector<CgalPointWithNormal>& points_with_normals)
{
	const int nb_neighbors = 18;
	double spacing = CGAL::compute_average_spacing<CgalConcurrencyTag>(
		points_with_normals, nb_neighbors,
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<CgalPointWithNormal>()));

	CGAL::pca_estimate_normals<CgalConcurrencyTag>(points_with_normals, nb_neighbors,
												   cgal_params.neighbor_radius(1. * spacing));

	// Naively orient normals towards the lidar (0, y, z)
	for (auto& point : points_with_normals) {
		auto& [p, n] = point;
		Eigen::Vector3d point_to_lidar{0, p.y(), p.z()};
		Eigen::Vector3d normal(0, n.y(), n.z());
		if (point_to_lidar.dot(normal) > 0) n *= -1;
	}
}
