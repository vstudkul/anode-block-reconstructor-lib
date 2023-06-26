/*
	Язык: C++17
	Используемые во время разработки библиотеки и их версии: 
		- libpcl 1.13 (Point Cloud Library, https://pointclouds.org/)
		- libcgal-dev 5.5.1-2 (Computational Geometry Algorithms Library, https://www.cgal.org/)
		- libopen3d-dev 0.16.1 (Open3D, http://www.open3d.org/)
		- libfmt 9.1.0 (fmt, https://fmt.dev/)
*/

#include <iostream>
#include <string>
#include <pcl/console/parse.h>

#include "anode_reconstructor_lib/src/anode_block_reconstructor.h"

int main(int argc, char** argv)
{
	std::string fileName;
	std::string second_half_filename;
	std::string output_path;

	// Считываем параметры для файлов со сканами первой и второй стенки, а также путь для выходного файла
	// с реконструированным мешем
	pcl::console::parse(argc, argv, "-p", fileName);
	pcl::console::parse(argc, argv, "--second-half", second_half_filename);
	pcl::console::parse_argument(argc, argv, "--output", output_path);

	// Параметры поворота облака точек ("выход" из системы координат лидара)
	float rotateX = 90, rotateY = 135, rotateZ = 90;
	double lidar_height = 1.525; // высота лидара (для отсечения конвеера)
	// расстояние между лидарами (приблизительно) для объединения половинок
	float distance_between_lidars = 2.8; 

	AnodeBlockReconstructor rec(fileName, second_half_filename);
	rec.RotateCloud(false, rotateX, rotateY, rotateZ);
	rec.RotateCloud(true, rotateX, rotateY, rotateZ);

	// Прореживаем сетку вокселей (опционально). Ускоряет общую скорость работы алгоритмов
	// за счёт уменьшения количества точек. Отрицательно сказывается на точности поиска повреждений.
	rec.DownsampleCloud(0.005);

	// Если блок стоит неровно на ленте, поворачиваем в сторону лидара (опционально, медленно)
	rec.OrientHalvesTowardsLidar();
	// Убираем поверхность конвеера, если известна высота (опционально)
	rec.CutOffConveyorBeltGround(lidar_height);

	// Ищем анодные блоки на двух сканах, объединяем их в один, достраиваем недостающие стенки и удаляем все лишние точки
	rec.ProcessAnodeBlockHalves(distance_between_lidars, false);

	// Получаем размер стенок блока (ширина и высота)
	auto size = rec.GetAnodeBlockSize();
	std::printf("Anode Block Wall size: X: %f, Y: %f\n", size.x(), size.y());
	
	// Реконструируем поверхность методом Пуассона
	rec.RunSurfaceReconstruction();

	// Ищем повреждения (помечаем вершины как правильные/неправильные).
	// Опционально можно запустить Region Growing для отсечения сегментов, 
	//	в которые входит слишком малое количество точек.
	// Опционально можно передать расстояние до эталонной плоскости, 
	//	ближе которого вершины никогда не будут отмечаться как повреждённые.
	rec.RunDefectDetection(false, 0.005);

	// Выводим результат поиска повреждений на стенках
	auto wall_a_defect = rec.GetFirstWallDefects();
	auto wall_b_defect = rec.GetSecondWallDefects();
	std::printf("Wall A) Damaged Points: %d/%d (%f)\n",
		wall_a_defect.number_of_damaged_points, 
		wall_a_defect.number_of_total_points, 
		wall_a_defect.percent_of_damaged_points);
	std::printf("Wall B) Damaged Points: %d/%d (%f)\n",
		wall_b_defect.number_of_damaged_points, 
		wall_b_defect.number_of_total_points, 
		wall_b_defect.percent_of_damaged_points);
		
	// Выводим реконструированный меш на экран с выделенными повреждениями
	rec.VisualizeMesh(true);

	// Сохраняем этот же меш в файл в PLY формате
	rec.SaveReconstructedMeshWithDefects(output_path);
}
