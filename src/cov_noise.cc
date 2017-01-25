// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_noise.h"
#include <cmath>

namespace libgp
{

CovNoise::CovNoise()
{
	param_dim = 1;
    loghyper.resize(param_dim);
    loghyper.setZero();
}

CovNoise::~CovNoise() {}

double CovNoise::get(const GPData& x1, const GPData& x2)
{
	if (&x1 != &x2) {
		return 0.0;
	}
	if (x1.info.size() == 0) {
		return s2;
	}
	return s2 * x1.info(0) * x1.info(0);
}

void CovNoise::grad(const GPData& x1, const GPData& x2, Eigen::VectorXd &grad)
{
	if (&x1 != &x2) {
		grad(0) = 0.0;
	} else if (x1.info.size() == 0) {
		grad(0) = 2 * s2;
	} else {
		grad(0) = 2 * s2 * x1.info(0) * x1.info(0);
	}
}

void CovNoise::set_loghyper(const Eigen::VectorXd &p)
{
    CovarianceFunction::set_loghyper(p);
    s2 = exp(2*loghyper(0));
}

std::string CovNoise::to_string()
{
    return "CovNoise";
}

double CovNoise::get_threshold()
{
    return 0.0;
}

void CovNoise::set_threshold(double threshold) {}

}
