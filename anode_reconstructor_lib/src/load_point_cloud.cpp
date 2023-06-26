#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>

#include "../lib/rapidcsv.h"
#include <fmt/core.h>

#include "load_point_cloud.hpp"
#include "reconstruction_utils.hpp"


static bool StringEndsWith(const std::string& s, const std::string& suffix)
{
	// eg. VEHICLE, CLE => 4 == 7 - 3
	return s.rfind(suffix) == s.length() - suffix.length();
}

CloudPtr ReadDataFromCSV(const std::string& filePath)
{
	pcl::PointCloud<Point>::Ptr pointCloud(new pcl::PointCloud<Point>);

	rapidcsv::Document doc;
	// extract value separator
	// println("Loading file '{}'", filePath);
	std::string header;
	std::ifstream file_stream(filePath);
	if (not file_stream.good() || filePath.empty()) {
		Println("== There is something wrong with this 'csv' file: '{}'", filePath);
		std::exit(-1);
	}

	std::getline(file_stream, header);

	char separator = '?';
	std::string possibleSeparators = ",; ";
	for (const auto& sep : possibleSeparators)
		if (header.find(sep) != std::string::npos) {
			separator = sep;
			break;
		}

	// println("== CSV Header: '{}'", header);
	// println("== Detected CSV-like file separator: '{}'", separator);
	// skip header and empty lines, treat NaN and empty cells as '-1'
	doc.Load(filePath, rapidcsv::LabelParams(), rapidcsv::SeparatorParams(separator, true),
			 rapidcsv::ConverterParams(true, -1), rapidcsv::LineReaderParams(false, '#', true));
	for (int i = 0; i < doc.GetRowCount(); ++i) {
		auto p = doc.GetRow<float>(i);
		Point point(p[0], p[1], p[2]);
		point.r = point.g = point.b = 128 + 64;
		pointCloud->points.push_back(point);
	}

	pointCloud->width = pointCloud->size();
	pointCloud->height = 1;

	return pointCloud;
}

CloudPtr LoadPointCloudFromFile(const std::string& fileName)
{
	CloudPtr cloud(new Cloud);
	if (StringEndsWith(fileName, "pcd")) {
		// load file as pcd
		pcl::io::loadPCDFile(fileName, *cloud);
		bool color_is_off = std::all_of(cloud->points.begin(), cloud->points.end(),
										[&cloud](auto& elem) { return elem.rgb == cloud->points[0].rgb; });
		if (color_is_off) {
			// std::cout << "Can't find the color segment in the scan. Coloring white..." << std::endl;
			for (auto& p : cloud->points)
				p.r = p.g = p.b = 128 + 64;
		}
	} else if (StringEndsWith(fileName, "ply")) {
		// load file as ply
		if (pcl::io::loadPLYFile(fileName, *cloud) == -1) {
			std::cout << "File not found!" << std::endl;
			std::exit(-1);
		}

		bool color_is_off = std::all_of(cloud->points.begin(), cloud->points.end(),
										[&cloud](auto& elem) { return elem.rgb == cloud->points[0].rgb; });
		if (color_is_off) {
			// std::cout << "Can't find the color segment in the scan. Coloring white..." << std::endl;
			for (auto& p : cloud->points)
				p.r = p.g = p.b = 128 + 64;
		}
	} else if (StringEndsWith(fileName, "stl")) {
		// load points from stl
		pcl::PolygonMesh mesh;
		pcl::io::loadPolygonFile(fileName, mesh);
		pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
		for (auto& p : cloud->points)
			p.r = p.g = p.b = 128 + 64;
	} else {
		// load as csv or ssv
		cloud = ReadDataFromCSV(fileName);
	}
	return cloud;
}
