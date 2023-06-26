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

#include "reconstruction.hpp"


void RunMeshVisualizer(const VisCtx& ctx)
{
	// auto coordinate_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame();

	if (not ctx.mesh_with_defects) {
		open3d::visualization::DrawGeometries({ctx.mesh_without_defects});
		return;
	}

	auto removeAll = [&](open3d::visualization::Visualizer* vis) {
		vis->RemoveGeometry(*ctx.wall_a_rect, false);
		vis->RemoveGeometry(*ctx.wall_b_rect, false);
		vis->RemoveGeometry(*ctx.mesh_with_defects, false);
		vis->RemoveGeometry(*ctx.reconstructed_block, false);
		vis->RemoveGeometry(ctx.mesh_without_defects, false);
	};

	const std::map<int, std::function<bool(open3d::visualization::Visualizer*)>> key_to_callback = {
		{'Z',
		 [&](open3d::visualization::Visualizer* vis) {
			 // hide defects
			 removeAll(vis);
			 vis->AddGeometry(ctx.mesh_without_defects, false);
			 return true;
		 }},
		{'X',
		 [&](open3d::visualization::Visualizer* vis) {
			 // show defects
			 removeAll(vis);
			 vis->AddGeometry(*ctx.wall_a_rect, false);
			 vis->AddGeometry(*ctx.wall_b_rect, false);
			 vis->AddGeometry(*ctx.mesh_with_defects, false);
			 return true;
		 }},
		{'C',
		 [&](open3d::visualization::Visualizer* vis) {
			 // show point cloud
			 removeAll(vis);
			 vis->AddGeometry(*ctx.reconstructed_block, false);
			 return true;
		 }},
		{'V',
		 [&](open3d::visualization::Visualizer* vis) {
			 // enter point selection / measure mode
			 open3d::visualization::DrawGeometriesWithVertexSelection({ctx.mesh_without_defects});
			 return true;
		 }},
	};


	open3d::visualization::DrawGeometriesWithKeyCallbacks({*ctx.mesh_with_defects, *ctx.wall_a_rect, *ctx.wall_b_rect},
														  key_to_callback);
}

double GetPointToPlaneDistance(const Eigen::Vector3d& point, Eigen::Vector3d& normal, double d)
{
	normal = normal.normalized();
	return std::abs(normal.x() * point.x() + normal.y() * point.y() + normal.z() * point.z() + d);
}

CloudWithNormalsPtr RunIcp(const CloudWithNormalsPtr target, const CloudWithNormalsPtr source)
{
	auto remove_z = [](CloudWithNormalsPtr cloud) {
		for (auto& p : cloud->points)
			p.z = 0;
	};

	double dist = 10;
	double rans = 0.05;
	int iter = 500;


	WarpPointRigid3DTrans<PointWithNormal, PointWithNormal>::Ptr warp_fcn(
		new WarpPointRigid3DTrans<PointWithNormal, PointWithNormal>);
	pcl::registration::TransformationEstimationLM<PointWithNormal, PointWithNormal>::Ptr te(
		new pcl::registration::TransformationEstimationLM<PointWithNormal, PointWithNormal>);
	te->setWarpFunction(warp_fcn);

	CloudWithNormalsPtr source_copy(new CloudWithNormals), target_copy(new CloudWithNormals);
	pcl::copyPointCloud(*target, *target_copy);
	pcl::copyPointCloud(*source, *source_copy);

	RunCloudDownsampling(target_copy, 0.03);
	RunCloudDownsampling(source_copy, 0.03);
	remove_z(target_copy);
	remove_z(source_copy);

	pcl::IterativeClosestPointNonLinear<PointWithNormal, PointWithNormal> icp;
	icp.setTransformationEstimation(te);
	icp.setMaximumIterations(iter);
	icp.setMaxCorrespondenceDistance(dist);
	icp.setRANSACOutlierRejectionThreshold(rans);
	icp.setInputTarget(target_copy);
	icp.setInputSource(source_copy);
	icp.align(*source_copy);

	CloudWithNormalsPtr result(new CloudWithNormals);
	pcl::transformPointCloud(*source, *result, icp.getFinalTransformation());

	return result;
}

void AlignScans(CloudWithNormalsPtr base, CloudWithNormalsPtr sub_mut, float lidar_dist, bool use_icp)
{
	// println("Before Orientation; Wall1: {}; Wall2: {}; Dist: {}", base->size(), sub_mut->size(), lidar_dist);

	// determine the direction where we need to move the second part
	pcl::PointXYZRGBNormal c1, c2;
	pcl::computeCentroid(*base, c1);
	pcl::computeCentroid(*sub_mut, c2);

	for (auto& p : sub_mut->points) {
		p.rgba = 0xFF000099;
		p.y *= -1;
		p.normal_y *= -1;

		// shift the second part along the Y axis to make the life a bit easier for the ICP algorithm
		float offset_y = use_icp ? 2.8 : 0;
		// float offset_y = 5;
		p.y += lidar_dist + offset_y;

		// (-0.500, 2.78, 0)
		p.x += (c1.x - c2.x);
	}
	// println("LidarYOffset: {}; Lidar2YOffset: {}", lidar_y_offset, lidar_y_offset_part2);

	if (use_icp) {
		auto new_cloud = RunIcp(base, sub_mut);
		for (int i = 0; i < sub_mut->points.size(); i++) {
			sub_mut->points[i] = new_cloud->points[i];
		}
	}

	// open3d::io::WritePointCloud("/tmp/randdir607302/A.ply", *pcl_to_open3d(base));
	// open3d::io::WritePointCloud("/tmp/randdir607302/B.ply", *pcl_to_open3d(sub_mut));
	// open3d::visualization::DrawGeometries({pcl_to_open3d(sub_mut), pcl_to_open3d(base)});
	// exit(0);
}

void RemoveConveyorBeltGround(CloudPtr cloud, double lidar_height)
{
	CloudPtr newPoints(new Cloud);

	// remove the ground
	for (auto& p : cloud->points) {
		if (p.z >= -lidar_height) {
			newPoints->points.push_back(p);
		}
	}

	cloud->points = newPoints->points;
}

AnodeBlockWall ProcessHalfOfAnodeBlock(CloudPtr point_cloud, int num, std::string debug_path)
{
	// println("Start 'process_half_of_anode_block'; Cloud: {}; num={}", point_cloud->size(), num);
	CGAL::get_default_random() = CGAL::Random(0);

	// Reconstuct normals
	auto points_with_normals = PclToCgal(point_cloud);
	ReconstructNormals(points_with_normals);
	if (not debug_path.empty())
		CGAL::IO::write_PLY(fmt::format("{}/009_pc_with_normals_{}.ply", debug_path, num), points_with_normals,
							cgal_params);

	// Find plane in the scan
	auto [wall_points_pcl, wall_norm, wall_d] = RunCgalRansac(points_with_normals);
	// println("Original pc plane: norm={}; d={}", _wall_norm, _wall_d);
	StatisticalOutlierRemoval(wall_points_pcl);

	if (not debug_path.empty())
		open3d::io::WritePointCloud(fmt::format("{}/010_wall_points_pcl_{}.ply", debug_path, num),
									*PclToOpen3d(wall_points_pcl));

	// Collect all points of the anode block
	auto [wall_bb_min, wall_bb_max] = GetAABoundingBox(wall_points_pcl);
	Eigen::Vector3d block_bb_min = {wall_bb_min.x(), wall_bb_min.y() - 0.1, wall_bb_min.z()};
	// Eigen::Vector3d block_bb_min = {wall_bb_min.x(), wall_bb_min.y() /* - 0.1 */, wall_bb_min.z()};
	// extend BB towards the block roof
	Eigen::Vector3d block_bb_max = {wall_bb_max.x(), wall_bb_max.y() + 0.75, // "0.75" is a block width along Y
									wall_bb_max.z() + 10}; // include roof? (normally 0.08)
	auto points_pcl = CgalToPcl(points_with_normals);
	StatisticalOutlierRemoval(points_pcl); // try to run
	// statisticalOutlierRemoval(points_pcl); // try to run

	auto anode_block_points = FilterByBoundingBox(points_pcl, block_bb_min, block_bb_max);

	// statisticalOutlierRemoval(anode_block_points); // try to run
	return AnodeBlockWall{.cloud = anode_block_points,
						  .normal = wall_norm,
						  .coeff_d = wall_d,
						  .bb_min = wall_bb_min,
						  .bb_max = wall_bb_max};
}

void GetWallAngleAndRotateAnodeBlock(CloudPtr cloud_a, CloudPtr cloud_b)
{
	std::vector<cv::Point2f> hough_points;
	for (auto& p : cloud_a->points)
		hough_points.push_back(cv::Point2f(p.x, p.y));
	std::vector<cv::Vec3d> hough_lines;
	cv::HoughLinesPointSet(hough_points, hough_lines, 3, 1000, 0, 2, 0.01, 0, 2 * M_PI, M_PI / 180 / 20);


	cv::Vec3d best_hough_line = hough_lines[0];
	auto [hough_line_dist, hough_line_angle] = std::tie(best_hough_line[1], best_hough_line[2]);
	double h_cos = cos(hough_line_angle), h_sin = sin(hough_line_angle);
	Eigen::Vector3d hough_normal = {h_cos, h_sin, 0};
	// println("Wall Detection: HOUGH: {}; Points: {}; Hough Normal: {}; Angle: {}", hough_lines, hough_points.size(),
	// 		hough_normal, (hough_line_angle * 180 / M_PI) - 90);


	double angle = hough_line_angle - (M_PI / 2);
	// println("Anode Block Wall Angle (rad): {}", angle);

	// Eigen::Affine3f transform_move_before = Eigen::Affine3f::Identity();
	// transform_move_before.translation() << -line_components[2], -line_components[3], 0.0;

	Eigen::Affine3f transform_rotate = Eigen::Affine3f::Identity();
	transform_rotate.rotate(Eigen::AngleAxisf(-angle, Eigen::Vector3f::UnitZ()));

	// Eigen::Affine3f transform_move_after = Eigen::Affine3f::Identity();
	// transform_move_after.translation() << line_components[2], line_components[3], 0.0;

	// pcl::transformPointCloud(*cloud, *cloud, transform_move_before);
	pcl::transformPointCloud(*cloud_a, *cloud_a, transform_rotate);
	pcl::transformPointCloud(*cloud_b, *cloud_b, transform_rotate);
	// pcl::transformPointCloud(*cloud, *cloud, transform_move_after);
}

DefectsSearchResult FindDefectsInOneWall(int wall_n,
										 Open3dMeshPtr poisson_mesh,
										 Eigen::Vector3d orig_wall_bb_max,
										 Eigen::Vector3d orig_wall_bb_min,
										 AABoundingBox full_block_bb,
										 CloudWithNormalsPtr wall_cloud,
										 double wall_dist_threshold)
{
	//
	// Convert the mesh back to the point cloud form
	//
	CloudWithNormalsPtr half_block_mesh_verts(new CloudWithNormals);
	auto block_center = (full_block_bb.max + full_block_bb.min) / 2;
	auto block_size = full_block_bb.max - full_block_bb.min;
	std::vector<Eigen::Vector3d> points_o3d;
	for (int i = 0; i < poisson_mesh->vertices_.size(); ++i) {
		points_o3d.push_back(poisson_mesh->vertices_[i]);
		auto& col = poisson_mesh->vertex_colors_[i];
		auto& pos = poisson_mesh->vertices_[i];
		auto& norm = poisson_mesh->vertex_normals_[i];
		if (wall_n == 0) {
			if (pos.y() < block_center.y() - block_size.y() / 3)
				half_block_mesh_verts->emplace_back(pos.x(), pos.y(), pos.z(), 1, 1, 1, norm.x(), norm.y(), norm.z());
		} else {
			if (pos.y() >= block_center.y() + block_size.y() / 3)
				half_block_mesh_verts->emplace_back(pos.x(), pos.y(), pos.z(), 1, 1, 1, norm.x(), norm.y(), norm.z());
		}
	}

	auto [reconstructed_wall_points_pcl, rec_norm, rec_d] = RunCgalRansac(PclToCgal(half_block_mesh_verts));

	auto reconstructed_wall_aabb = GetAABoundingBox(reconstructed_wall_points_pcl);
	auto reconstructed_wall_points_aabb =
		std::make_shared<Open3dAABB>(reconstructed_wall_aabb.min, reconstructed_wall_aabb.max);
	reconstructed_wall_points_aabb->color_ = {1, 0, 0};

	auto indices_o3d = reconstructed_wall_points_aabb->GetPointIndicesWithinBoundingBox(points_o3d);


	//
	//
	// Make 2D Hough plane
	//
	std::vector<cv::Point2f> hough_points;
	for (auto& p : wall_cloud->points) {
		hough_points.push_back(cv::Point2f(p.x, p.y));
	}
	std::vector<cv::Vec3d> hough_lines;
	cv::HoughLinesPointSet(hough_points, hough_lines, 3, 100, 0, 10, 0.0004, 0, M_PI, M_PI / 180 / 10);

	cv::Vec3d best_hough_line = hough_lines[0];
	auto [hough_line_dist, hough_line_angle] = std::tie(best_hough_line[1], best_hough_line[2]);
	double h_cos = cos(hough_line_angle), h_sin = sin(hough_line_angle);
	Eigen::Vector3d hough_normal = {h_cos, h_sin, 0};


	//
	// Draw hough plane
	//

	double p_x0 = full_block_bb.min.x() - 0.1, p_x1 = full_block_bb.max.x() + 0.1;
	double p_z0 = full_block_bb.min.z() - 0.1, p_z1 = full_block_bb.max.z();
	// Ax + By + C = 0
	double p_y0 = (p_x0 * hough_normal.x() - hough_line_dist) / (-hough_normal.y());
	double p_y1 = (p_x1 * hough_normal.x() - hough_line_dist) / (-hough_normal.y());

	std::vector<Eigen::Vector3d> hough_points_o3d = {
		{p_x1, p_y1, p_z0}, {p_x1, p_y1, p_z1}, {p_x0, p_y0, p_z1}, {p_x0, p_y0, p_z0}};

	// auto hough_wall_lineset = std::make_shared<open3d::geometry::LineSet>(
	Open3dLineSetPtr hough_wall_lineset(
		new Open3dLineSet(hough_points_o3d, std::vector<Eigen::Vector2i>{{0, 1}, {1, 2}, {2, 3}, {3, 0}}));
	hough_wall_lineset->PaintUniformColor({1, 0, 0});


	//
	// Mark damaged regions
	//

	auto checkIfPointBelongsToWall = [&](auto p) {
		auto bb_dist = orig_wall_bb_max - orig_wall_bb_min;
		return p.x() >= orig_wall_bb_min.x() + bb_dist.x() * 0.01 &&
			   p.x() <= orig_wall_bb_max.x() - bb_dist.x() * 0.01 &&
			   p.z() >= orig_wall_bb_min.z() + bb_dist.z() * 0.01 && p.z() <= orig_wall_bb_max.z() - bb_dist.z() * 0.03;
	};

	double wall_max_dist = 0;
	double wall_average_dist;
	int n_wall_points = 0;
	for (const auto& index : indices_o3d) {
		double dist = GetPointToPlaneDistance(poisson_mesh->vertices_[index], hough_normal, -hough_line_dist);
		auto& p = poisson_mesh->vertices_[index];
		if (checkIfPointBelongsToWall(p)) {
			n_wall_points += 1;
			wall_max_dist = std::max(wall_max_dist, dist);
			wall_average_dist += dist;
		}
	}
	wall_average_dist /= n_wall_points;


	auto wall_mesh_points_o3d = std::make_shared<open3d::geometry::PointCloud>();

	double mse = 0;
	// color damaged regions
	int n_damaged = 0;
	double dmg_score = 0;
	double defect_dist_threshold = wall_average_dist * 2.8; // 3.14; // 2.4; //* 3.14;

	defect_dist_threshold = std::max(wall_dist_threshold, defect_dist_threshold);
	for (const auto& index : indices_o3d) {
		double dist = GetPointToPlaneDistance(poisson_mesh->vertices_[index], hough_normal, -hough_line_dist);

		auto& p = poisson_mesh->vertices_[index];
		if (checkIfPointBelongsToWall(p)) {
			wall_mesh_points_o3d->points_.push_back(p);

			mse += pow(dist - wall_average_dist, 2);

			if (dist > defect_dist_threshold) {
				double x = (dist - defect_dist_threshold) / (wall_max_dist - defect_dist_threshold);
				poisson_mesh->vertex_colors_[index] = {x, 0, 0};
				n_damaged++;
				dmg_score += dist - defect_dist_threshold;
				wall_mesh_points_o3d->colors_.push_back({1, 0, 0});
			} else {
				wall_mesh_points_o3d->colors_.push_back({0, 0, 0});
			}
		}
	}
	mse /= n_wall_points;


	// println("== Wall {}) Average Distance to Ideal Plane: {:.12f}; Max dist: {:.12f}; MSE: {:.12f}; RMSE: {:.12f}",
	// 		wall_n, wall_average_dist, wall_max_dist, mse, sqrt(mse));
	// println("== Wall {}) Points Considered Damaged {}/{} ({}%); BadnessScore: {}", wall_n, n_damaged, n_wall_points,
	// 		(n_damaged * 100 / n_wall_points), dmg_score);

	DefectsSearchResult res;
	res.lineset = hough_wall_lineset;
	res.average_distance_to_wall = wall_average_dist;
	res.mse = mse;
	res.rmse = sqrt(mse);
	res.number_of_damaged_points = n_damaged;
	res.percent_of_damaged_points = (n_damaged * 100.0) / n_wall_points;
	res.number_of_total_points = n_wall_points;
	return res;
}

void ApplyRegionGrowing(Open3dMeshPtr mesh)
{
	CloudPtr cloud(new Cloud);
	for (int i = 0; i < mesh->vertices_.size(); i++) {
		auto point = mesh->vertices_[i];
		auto color = mesh->vertex_colors_[i];
		cloud->push_back(Point(point.x(), point.y(), point.z(), color.x() * 255, color.y() * 255, color.z() * 255));
	}

	pcl::search::Search<Point>::Ptr tree(new pcl::search::KdTree<Point>);
	pcl::IndicesPtr indices(new std::vector<int>);
	pcl::removeNaNFromPointCloud(*cloud, *indices);

	pcl::RegionGrowingRGB<Point> reg;
	reg.setInputCloud(cloud);
	reg.setIndices(indices);
	reg.setSearchMethod(tree);
	reg.setDistanceThreshold(0.01);
	reg.setPointColorThreshold(150);
	reg.setRegionColorThreshold(100);
	reg.setMinClusterSize(100);

	std::vector<pcl::PointIndices> clusters;
	reg.extract(clusters);
	CloudPtr colored_cloud = reg.getColoredCloud();

	std::sort(clusters.begin(), clusters.end(),
			  [](auto& a, auto& b) { return (int64_t)a.indices.size() > (int64_t)b.indices.size(); });

	// for (int i = 0; i < clusters.size(); i++) {
	// 	println("ClusterSize: {}; {}", i, clusters[i].indices.size());
	// }

	auto biggest_cluster = clusters[0];

	for (auto& p : colored_cloud->points) {
		p.r = 255;
		p.g = 0;
		p.b = 0;
	}
	for (auto i : biggest_cluster.indices) {
		auto& p = colored_cloud->points[i];
		p.r = 128;
		p.g = 128;
		p.b = 128;
	}

	for (int i = 0; i < colored_cloud->points.size(); i++) {
		auto& p = colored_cloud->points[i];
		auto& col = mesh->vertex_colors_[i];
		col.x() = (float)p.r / 255.0;
		col.y() = (float)p.g / 255.0;
		col.z() = (float)p.b / 255.0;
	}
}
