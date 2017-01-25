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
    int n=20, m=20;
    double tss = 0, error, f, y;
    // initialize Gaussian process for 2-D input using the squared exponential
    // covariance function with additive white noise.

    GaussianProcess gp(1, 1, "CovSum ( CovSEiso, CovNoise)");
    // initialize hyper parameter vector
    Eigen::VectorXd params(gp.covf().get_param_dim());
    params << 0.0, 0.0, -2.0;
    // set parameters of covariance function
    gp.covf().set_loghyper(params);
    // add training patterns
#if 1
    PointCloudColor::Ptr training(new PointCloudColor());
    for(int i = 0; i < n; ++i) {
        double x = drand48() * 10;
		y = sin(x) + Utils::randn() * 0.02;
        gp.add_pattern(&x, y);

        pcl::PointXYZRGBA pt;
        pt.x = x;
        pt.y = 0.0;
        pt.z = y;
        pt.rgba = 0xFF00FF00;
        training->points.push_back(pt);
    }
    training->width = 1;
    training->height = training->points.size();

    PointCloudColor::Ptr testing(new PointCloudColor());
    // total squared error
    for(int i = 0; i < m; ++i) {
        double x = drand48() * 10;
        f = gp.f(&x);
        y = sin(x);
        error = f - y;
        tss += error*error;

        pcl::PointXYZRGBA pt;
        pt.x = x;
        pt.y = 0.0;
        pt.z = y;
        pt.rgba = 0xFFFFFF00;
        testing->points.push_back(pt);
    }
    testing->width = 1;
    testing->height = testing->points.size();

    std::cout << "mse = " << tss/m << std::endl;
    pcl::visualization::PCLVisualizer viewer;
    viewer.setBackgroundColor(0, 0, 0);
    viewer.addCoordinateSystem(2.0);
    viewer.initCameraParameters();
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(training);
	viewer.addPointCloud<pcl::PointXYZRGBA>(training, rgb, "training");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "training");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb1(testing);
	viewer.addPointCloud<pcl::PointXYZRGBA>(testing, rgb1, "testing");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "testing");
    while (!viewer.wasStopped()) {
        viewer.spinOnce(10);
    }
#endif
    ProfilerStop();
    return EXIT_SUCCESS;
}
