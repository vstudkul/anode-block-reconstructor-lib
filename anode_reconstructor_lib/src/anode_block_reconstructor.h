#pragma once

#include <optional>
#include <string>

#include <pcl/common/io.h>
#include <pcl/point_cloud.h>

#include "load_point_cloud.hpp"
#include "reconstruction.hpp"
#include "types.hpp"


class AnodeBlockReconstructor {
	CloudPtr block_part_a;
	CloudPtr block_part_b;

	float distance_between_lidars;
	float lidar_height;
	float init_downsample_level;
	std::string debug_path;

	Eigen::Vector3d block_size;
	Open3dCloudPtr reconstructed_block_points;
	Open3dMeshPtr reconstructed_block_mesh;
	Open3dMeshPtr reconstructed_block_mesh_with_defects;

	VisCtx vis_ctx;
	AnodeBlockWall wall_a;
	AnodeBlockWall wall_b;
	AABoundingBox block_bb;

	DefectsSearchResult defects_wall_a;
	DefectsSearchResult defects_wall_b;


public:
	//=============================================================================
	// Constructors
	//=============================================================================

	/**
	* @brief Создаёт главный объект для реконструкции из двух облаков точек в виде объектов из библиотеки Point Cloud Library
	*
	* @param _block_part_a Облако точек для первой половины блока
	* @param _mb_block_part_b Облако точек для второй половины блока
	*/
	AnodeBlockReconstructor(CloudPtr _block_part_a, std::optional<CloudPtr> _mb_block_part_b = {});

	/**
	* @brief Создаёт главный объект для реконструкции из двух облаков точек в виде путей до csv-файлов
	* 
	* @param scan_a_path Путь для первого csv-подобного текстового файла
	* @param mb_scan_b_path Путь до второго csv-подобного текстового файла
	*/
	AnodeBlockReconstructor(std::string scan_a_path, std::optional<std::string> mb_scan_b_path = {});

	//=============================================================================
	// Getters/Setters
	//=============================================================================

	/**
	* @brief Задаёт путь для отладки, куда будут сохраняться промежуточные результаты работы ПО
	*/
	void SetDebugPath(std::string p);

	/**
	* @brief Возвращает размер блока вдоль осей X, Y и Z (ширина, глубина, высота)
	*/
	Eigen::Vector3d GetAnodeBlockSize();

	/**
	* @brief Возвращает облако точек для всего блока целиком
	*/
	Open3dCloudPtr GetReconstructedCloud();

	/**
	* @brief Возвращает реконструированный меш без отмеченных дефектов
	*/
	Open3dMeshPtr GetReconstructedSurface();

	/**
	* @brief Возвращает реконструированный меш с помеченными красным цветом дефектами
	*/
	Open3dMeshPtr GetReconstructedSurfaceWithFoundDefects();

	/**
	* @brief Возвращает облако точек для первой стены
	*/
	CloudPtr GetFirstHalfCloud();

	/**
	* @brief Возвращает облако точек для второй стены
	*/
	CloudPtr GetSecondHalfCloud();

	/**
	* @brief Повреждения на первой стенке
	*/
	DefectsSearchResult GetFirstWallDefects();

	/**
	* @brief Повреждения на второй стенке
	*/
	DefectsSearchResult GetSecondWallDefects();

	//=============================================================================
	// Main Routines
	//=============================================================================

	/**
	* @brief Вращаем облако точек (либо первой стенки, либо второй)
	* 
	* @param second_half Какую из половинок блока вращать
	* @param rotateX Угол вращения вокруг оси X
	* @param rotateY Угол вращения вокруг оси Y
	* @param rotateZ Угол вращения вокруг оси Z
	*/
	void RotateCloud(bool second_half, float rotateX, float rotateY, float rotateZ);

	/**
	* @brief Прореживание облака точек (уменьшение их количества для ускорения работы алгоритмов)
	* 
	* @param downsample_level Размер вокселя, в рамках которого окажется только 
	*/
	void DownsampleCloud(float downsample_level);

	/**
	* @brief Разворачиваем облака точек в сторону лидара (в случае, если анодный блок стоит не ровно на конвеере, медленно)
	*/
	void OrientHalvesTowardsLidar();

	/**
	* @brief Удаляет точки принадлежащие конвеерной ленте под блоком
	* 
	* @param lidar_height Высота над уровнем конвеера, на которой располагаются лидары
	*/
	void CutOffConveyorBeltGround(float lidar_height);

	/**
	* @brief Обработка каждой половины блока: выделение половинок блока на сканах, 
	*	объединение их в один, достройка недостающих стенок (слева, справа и снизу), удаление лишних точек.
	* 
	* @param distance_between_lidars Расстояние между лидарами
	* @param apply_icp Применять ли Iterative Closest Point алгоритм для объединения половинок блока, 
	*	если неизвестно расстояние (работает плохо)
	*/
	void ProcessAnodeBlockHalves(float distance_between_lidars, bool apply_icp);

	/**
	* @brief Реконструировать поверхность из облака точек методом Пуассона
	*/
	void RunSurfaceReconstruction();

	/**
	* @brief Запустить алгоритм обнаружения дефектов. 
	*	Строит эталонную плоскость, по расстонию до которой определяется наличие аномалии.
	* 
	* @param use_region_growing Применять ли алгоритм Region Growing для отсечения сегментов
	*	повреждённых точек со слишком малым количеством точек
	* @param wall_dist_threshold Расстояние до эталонной плоскости, меньше которого повреждения учитываться не будут
	*/
	void RunDefectDetection(bool use_region_growing, double wall_dist_threshold = 0.005);

	/**
	* @brief Визулировать реконструированный меш с выделенными повреждениями.
	* 	Кнопка 'Z' - показывает просто меш без повреждений
	* 	Кнопка 'X' - показывает меш с выделенными повреждениями
	*  Кнопка 'C' - показать исходное облако точек блока
	* 
	* @param show_defects Показывать ли дефекты
	*/
	void VisualizeMesh(bool show_defects);

	/**
	* @brief Сохранить реконструированный меш в файл в PLY формате (с отмеченными дефектами)
	* 
	* @param path Путь до файла, куда нужно сохранить меш
	*/
	void SaveReconstructedMesh(std::string path);

	/**
	* @brief Сохранить реконструированный меш в файл в PLY формате (без отмеченных дефектов)
	* 
	* @param path Путь до файла, куда нужно сохранить меш
	*/
	void SaveReconstructedMeshWithDefects(std::string path);
};