// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_se_ard.h"
#include <cmath>

namespace libgp
{
  
  CovSEard::CovSEard() {}

  CovSEard::CovSEard(const Eigen::VectorXd& p)
  {
	  assert(p.size() >= 2);
	  loghyper.resize(p.size());
	  CovarianceFunction::set_loghyper(p);
	  param_dim = p.size();
	  input_dim = p.size() - 1;
	ell.resize(input_dim);
    for(size_t i = 0; i < input_dim; ++i) ell(i) = exp(loghyper(i));
    sf2 = exp(2*loghyper(input_dim));
  }

  
  CovSEard::~CovSEard() {}
  
    double CovSEard::get(const GPData& x1, const GPData& x2)
  {  
    double z = (x1.x-x2.x).cwiseQuotient(ell).squaredNorm();
    return sf2*exp(-0.5*z);
  }
  
    void CovSEard::grad(const GPData& x1, const GPData& x2, Eigen::VectorXd &grad)
  {
    Eigen::VectorXd z = (x1.x-x2.x).cwiseQuotient(ell).array().square();  
    double k = sf2*exp(-0.5*z.sum());
    grad.head(input_dim) = z * k;
    grad(input_dim) = 2.0 * k;
  }
  
  void CovSEard::set_loghyper(const Eigen::VectorXd &p)
  {
	  assert(p.size() >= 2);
    loghyper.resize(p.size());
    CovarianceFunction::set_loghyper(p);
	input_dim = p.size() - 1;
	ell.resize(input_dim);
    for(size_t i = 0; i < input_dim; ++i) ell(i) = exp(loghyper(i));
    sf2 = exp(2*loghyper(input_dim));
  }
  
  std::string CovSEard::to_string()
  {
    return "CovSEard";
  }
}

