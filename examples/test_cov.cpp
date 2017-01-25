// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_noise.h"
#include "cov_se_iso.h"
#include "cov_sum.h"
#include "gp.h"
#include <gperftools/profiler.h>
#include <Eigen/Dense>
#include "sampleset.h"

using namespace libgp;

int main (int argc, char const *argv[])
{
    ProfilerStart("/tmp/gp.prof");
	CovarianceFunction::Ptr covf1(new CovNoise());
	Eigen::VectorXd params(covf1->get_param_dim());
	params << -2.0;
	covf1->set_loghyper(params);
	std::cout << "noise param: " << covf1->get_loghyper() << std::endl;
	GPData gpd1 = {
		.x = Eigen::Vector2d(1.0, 2.0),
		.info = Eigen::VectorXd(0),
		.y = 1.0
	};

	GPData gpd2 = {
		.x = Eigen::Vector2d(1.0, 2.0),
		.info = Eigen::VectorXd(0),
		.y = 1.0
	};

	std::cout << "cov_noise " << covf1->get(gpd1, gpd1) << std::endl;

	CovarianceFunction::Ptr covf2(new  CovSEiso());
	Eigen::VectorXd se_params(covf2->get_param_dim());
	se_params << 0.0, 0.0;
	covf2->set_loghyper(se_params);
	std::cout << "se " << covf2->get(gpd1, gpd2) << std::endl;

	CovarianceFunction::Ptr covf3(new CovSum());
	covf3->compound(covf2);
	covf3->compound(covf1);
	std::cout << "sum param dim: " << covf3->get_param_dim() << std::endl;
	std::cout << "sum param: " << covf3->get_loghyper() << std::endl;
	std::cout << "sum " << covf3->get(gpd1, gpd1) << std::endl;

	GaussianProcess gp;
	gp.set_cost_func(covf3);
	for (size_t i = 0; i < 2000; i++) {
		double x = 10 * drand48();
		double y = 10 * drand48();
		GPData gpd = {
			.x = Eigen::Vector2d(x, y),
			.info = Eigen::VectorXd(0),
			.y = sin(x) * sin(y)
		};
		gp.add_pattern(gpd);
	}
	GPData gpd3 = {
		.x = Eigen::Vector2d(5.0, 5.0),
		.info = Eigen::VectorXd(0),
		.y = 0.0
	};
	double f,var;
	gp.evaluate(gpd3, f, var);
	std::cout << "f " << f << " var " << var << std::endl;

    ProfilerStop();
    return 0;
}
