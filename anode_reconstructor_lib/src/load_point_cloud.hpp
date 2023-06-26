#pragma once

#include <pcl/common/angles.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include "types.hpp"

CloudPtr ReadDataFromCSV(const std::string& filePath);
CloudPtr LoadPointCloudFromFile(const std::string& fileName);
