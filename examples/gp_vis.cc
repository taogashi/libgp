// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "gp_utils.h"
#include <gperftools/profiler.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>

using namespace libgp;
using PointCloudColor = pcl::PointCloud<pcl::PointXYZRGBA>;

int main (int argc, char const *argv[])
{
    ProfilerStart("/tmp/gp.prof");

	GPData gpd1 = {
		.x = Eigen::Vector2d(2.0, 2.0),
		.info = Eigen::VectorXd(1),
		.y = 10.0
	};
	gpd1.info << 1.0;

	GPData gpd2 = {
		.x = Eigen::Vector2d(7.0, 7.0),
		.info = Eigen::VectorXd(1),
		.y = 10.0
	};
	gpd2.info << 1.0;

	Eigen::VectorXd params(1);
	params << -1.0;
	CovarianceFunction::Ptr covf1(new CovNoise(params));

	CovarianceFunction::Ptr covf2(new CovSEard(Eigen::Vector3d(2.0, -2.0, 0.5)));

	CovarianceFunction::Ptr covf3(new CovSum());
	covf3->compound(covf2);
	covf3->compound(covf1);

	GaussianProcess gp;
	gp.set_cost_func(covf3);

    int n=2000, m=2000;
    PointCloudColor::Ptr training(new PointCloudColor());
	for (size_t i = 0; i < n; i++) {
		double x = 10 * drand48() - 5;
		double y = 10 * drand48() - 5;
		GPData gpd = {
			.x = Eigen::Vector2d(x, y),
			.info = Eigen::VectorXd(1),
			.y = 0.4 * drand48() - 0.2
		};
		gpd.info << 5.0;
		gp.add_pattern(gpd);

        pcl::PointXYZRGBA pt;
        pt.x = x;
        pt.y = y;
        pt.z = gpd.y;
        pt.rgba = 0xFF00FF00;
        training->points.push_back(pt);
	}
	// add some outliers
#if 0
	{
		pcl::PointXYZRGBA pt;
		gp.add_pattern(gpd1);
		pt.x = gpd1.x(0);
		pt.y = gpd1.x(1);
		pt.z = gpd1.y;
        pt.rgba = 0xFF00FF00;
        training->points.push_back(pt);
		gp.add_pattern(gpd2);
		pt.x = gpd2.x(0);
		pt.y = gpd2.x(1);
		pt.z = gpd2.y;
        pt.rgba = 0xFF00FF00;
        training->points.push_back(pt);
	}
#endif
    training->width = 1;
    training->height = training->points.size();

    PointCloudColor::Ptr testing(new PointCloudColor());
	testing->points.resize(m*3);
#pragma omp parallel for
	for (size_t i = 0; i < m; i++) {
		double x = 20 * drand48() - 10;
		double y = 20 * drand48() - 10;
		GPData gpd = {
			.x = Eigen::Vector2d(x, y),
			.info = Eigen::VectorXd(1),
			.y = 0.0
		};
		gpd.info << 5.0;
		double f, var;
		gp.evaluate(gpd, f, var);

        testing->points[i].x = x;
        testing->points[i].y = y;
        testing->points[i].z = f;
        testing->points[i].rgba = 0xFFFFFF00;

		testing->points[m + i] = testing->points[i];
		testing->points[m + i].z = var;
		testing->points[m + i].rgba = 0x9FFF9F00;

		testing->points[2 * m + i] = testing->points[i];
		testing->points[2 * m + i].z = -var;
		testing->points[2 * m + i].rgba = 0x9FFF9F00;
    }
    testing->width = 1;
    testing->height = testing->points.size();

	std::cout << "after training: "
		<< gp.covf()->get_loghyper() << std::endl;
    pcl::visualization::PCLVisualizer viewer;
    viewer.setBackgroundColor(0, 0, 0);
    viewer.addCoordinateSystem(2.0);
    viewer.initCameraParameters();
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(training);
	viewer.addPointCloud<pcl::PointXYZRGBA>(training, rgb, "training");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "training");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb1(testing);
	viewer.addPointCloud<pcl::PointXYZRGBA>(testing, rgb1, "testing");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "testing");
    while (!viewer.wasStopped()) {
        viewer.spinOnce(10);
    }
    ProfilerStop();
    return EXIT_SUCCESS;
}
