#include <future>
#include <thread>

#include "anode_block_reconstructor.h"
#include "types.hpp"


AnodeBlockReconstructor::AnodeBlockReconstructor(CloudPtr _block_part_a, std::optional<CloudPtr> _mb_block_part_b)
{
	block_part_a.reset(new Cloud);
	pcl::copyPointCloud(*_block_part_a, *block_part_a);
	block_part_b.reset(new Cloud);
	if (_mb_block_part_b) {
		pcl::copyPointCloud(*_mb_block_part_b->get(), *block_part_b);
	} else {
		pcl::copyPointCloud(*_block_part_a, *block_part_b);
	}
}

AnodeBlockReconstructor::AnodeBlockReconstructor(std::string scan_a_path, std::optional<std::string> mb_scan_b_path)
{
	block_part_a = LoadPointCloudFromFile(scan_a_path);
	if (mb_scan_b_path) {
		block_part_b = LoadPointCloudFromFile(*mb_scan_b_path);
	} else {
		block_part_b = LoadPointCloudFromFile(scan_a_path);
	}
}

void AnodeBlockReconstructor::SetDebugPath(std::string p)
{
	debug_path = p;
}

Eigen::Vector3d AnodeBlockReconstructor::GetAnodeBlockSize()
{
	return block_size;
}

Open3dCloudPtr AnodeBlockReconstructor::GetReconstructedCloud()
{
	return reconstructed_block_points;
}

Open3dMeshPtr AnodeBlockReconstructor::GetReconstructedSurface()
{
	return reconstructed_block_mesh;
}

Open3dMeshPtr AnodeBlockReconstructor::GetReconstructedSurfaceWithFoundDefects()
{
	return reconstructed_block_mesh_with_defects;
}

CloudPtr AnodeBlockReconstructor::GetFirstHalfCloud()
{
	return block_part_a;
}

CloudPtr AnodeBlockReconstructor::GetSecondHalfCloud()
{
	return block_part_b;
}

DefectsSearchResult AnodeBlockReconstructor::GetFirstWallDefects()
{
	return defects_wall_a;
}

DefectsSearchResult AnodeBlockReconstructor::GetSecondWallDefects()
{
	return defects_wall_b;
}

void AnodeBlockReconstructor::RotateCloud(bool second_half, float rotateX, float rotateY, float rotateZ)
{
	auto cloud = second_half ? block_part_b : block_part_a;
	RotatePointCloud(cloud, rotateX, 0, 0);
	RotatePointCloud(cloud, 0, rotateY, 0);
	RotatePointCloud(cloud, 0, 0, rotateZ);
}

void AnodeBlockReconstructor::DownsampleCloud(float downsample_level)
{
	RunCloudDownsampling(block_part_a, downsample_level);
	RunCloudDownsampling(block_part_b, downsample_level);
}

void AnodeBlockReconstructor::OrientHalvesTowardsLidar()
{
	GetWallAngleAndRotateAnodeBlock(block_part_a, block_part_b);
}

void AnodeBlockReconstructor::CutOffConveyorBeltGround(float lidar_height)
{
	RemoveConveyorBeltGround(block_part_a, lidar_height);
	RemoveConveyorBeltGround(block_part_b, lidar_height);
}

void AnodeBlockReconstructor::ProcessAnodeBlockHalves(float distance_between_lidars, bool apply_icp)
{
	// parallelize a little bit
	std::future<AnodeBlockWall> wall_a_future =
		std::async(std::launch::async, [this]() { return ProcessHalfOfAnodeBlock(block_part_a, 0, debug_path); });

	wall_b = ProcessHalfOfAnodeBlock(block_part_b, 1, debug_path);
	wall_a = wall_a_future.get();

	if (not debug_path.empty()) {
		open3d::io::WritePointCloud(debug_path + "/020_anode_block_points_part1.ply", *PclToOpen3d(wall_a.cloud));
		open3d::io::WritePointCloud(debug_path + "/030_anode_block_points_part2.ply", *PclToOpen3d(wall_b.cloud));
	}

	AlignScans(wall_a.cloud, wall_b.cloud, distance_between_lidars, apply_icp);

	// construct full anode block
	CloudWithNormalsPtr anode_block(new CloudWithNormals);
	pcl::concatenate(*wall_a.cloud, *wall_b.cloud, *anode_block);

	if (not debug_path.empty()) {
		open3d::io::WritePointCloud(debug_path + "/040_almost_full_anode_block.ply", *PclToOpen3d(anode_block));
	}

	// bounding box for concatenated anode block
	auto block_bb = GetAABoundingBox(anode_block);
	auto new_anode_block_bb_max = Eigen::Vector3d{block_bb.max.x(), block_bb.max.y(), wall_b.bb_max.z()};
	block_size = new_anode_block_bb_max - block_bb.min;

	this->block_bb = block_bb;
	AddMissingWalls(anode_block, wall_a.bb_max, block_bb.min, block_bb.max);
	reconstructed_block_points = PclToOpen3d(anode_block);

	if (not debug_path.empty()) {
		open3d::io::WritePointCloud(debug_path + "/050_full_anode_block_without_missing_walls.ply",
									*reconstructed_block_points);
	}
}

void AnodeBlockReconstructor::RunSurfaceReconstruction()
{
	auto [mesh, _densities] = Open3dMesh::CreateFromPointCloudPoisson(*reconstructed_block_points);
	mesh->ComputeVertexNormals();
	mesh->PaintUniformColor({0.5, 0.5, 0.5});

	reconstructed_block_mesh = mesh;
	if (not debug_path.empty()) {
		open3d::io::WriteTriangleMesh(debug_path + "/060_open3d_poisson.ply", *reconstructed_block_mesh);
	}
}

void AnodeBlockReconstructor::RunDefectDetection(bool use_region_growing, double wall_dist_threshold)
{
	Open3dMeshPtr mesh_with_defects = std::make_shared<Open3dMesh>(*reconstructed_block_mesh);

	std::future<DefectsSearchResult> wall_a_future =
		std::async(std::launch::async, [this, wall_dist_threshold, mesh_with_defects]() {
			return FindDefectsInOneWall(0, mesh_with_defects, wall_a.bb_max, wall_a.bb_min, block_bb, wall_a.cloud,
										wall_dist_threshold);
		});

	auto wall_b_defects = FindDefectsInOneWall(1, mesh_with_defects, wall_a.bb_max, wall_a.bb_min, block_bb,
											   wall_b.cloud, wall_dist_threshold);
	auto wall_a_defects = wall_a_future.get();


	vis_ctx.mesh_without_defects = reconstructed_block_mesh;
	vis_ctx.mesh_with_defects = mesh_with_defects;
	vis_ctx.wall_a_rect = wall_a_defects.lineset;
	vis_ctx.wall_b_rect = wall_b_defects.lineset;

	defects_wall_a = wall_a_defects;
	defects_wall_b = wall_b_defects;

	if (not debug_path.empty()) {
		open3d::io::WriteTriangleMesh(debug_path + "/060_open3d_poisson_with_defects.ply", *mesh_with_defects);
	}
	if (use_region_growing) {
		ApplyRegionGrowing(mesh_with_defects);
		if (not debug_path.empty()) {
			open3d::io::WriteTriangleMesh(debug_path + "/060_open3d_poisson_with_defects_after_region_growing.ply",
										  *mesh_with_defects);
		}
	}

	this->reconstructed_block_mesh_with_defects = mesh_with_defects;
}

void AnodeBlockReconstructor::VisualizeMesh(bool show_defects)
{
	if (not show_defects) {
		vis_ctx.mesh_without_defects = {};
		vis_ctx.wall_a_rect = {};
		vis_ctx.wall_b_rect = {};
	}
	vis_ctx.reconstructed_block = reconstructed_block_points;
	RunMeshVisualizer(vis_ctx);

	if (not debug_path.empty()) {
		open3d::io::WritePointCloud(debug_path + "/070_open3d_points.ply", *reconstructed_block_points);
	}
}

void AnodeBlockReconstructor::SaveReconstructedMeshWithDefects(std::string path)
{
	open3d::io::WriteTriangleMesh(path, *reconstructed_block_mesh_with_defects);
}

void AnodeBlockReconstructor::SaveReconstructedMesh(std::string path)
{
	open3d::io::WriteTriangleMesh(path, *reconstructed_block_mesh);
}
