// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_se_iso.h"
#include <cmath>

namespace libgp
{

CovSEiso::CovSEiso()
{
    param_dim = 2;
    loghyper.resize(param_dim);
    loghyper.setZero();
}

CovSEiso::CovSEiso(const Eigen::VectorXd& p)
{
	assert(p.size() == 2);
    param_dim = 2;
	loghyper.resize(2);
	CovarianceFunction::set_loghyper(p);
    ell = exp(loghyper(0));
    sf2 = exp(2*loghyper(1));
}

CovSEiso::~CovSEiso() {}

double CovSEiso::get(const GPData& x1, const GPData& x2)
{
    double z = ((x1.x-x2.x)/ell).squaredNorm();
    return sf2*exp(-0.5*z);
}

void CovSEiso::grad(const GPData& x1, const GPData& x2, Eigen::VectorXd &grad)
{
    double z = ((x1.x-x2.x)/ell).squaredNorm();
    double k = sf2*exp(-0.5*z);
    grad << k*z, 2*k;
}

void CovSEiso::set_loghyper(const Eigen::VectorXd &p)
{
    CovarianceFunction::set_loghyper(p);
    ell = exp(loghyper(0));
    sf2 = exp(2*loghyper(1));
}

std::string CovSEiso::to_string()
{
    return "CovSEiso";
}

}
