// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov.h"
#include "gp_utils.h"

namespace libgp
{
  
  size_t CovarianceFunction::get_param_dim()
  {
    return param_dim;
  }
  
  Eigen::VectorXd CovarianceFunction::get_loghyper()
  {
    return loghyper;
  }
  
  void CovarianceFunction::set_loghyper(const Eigen::VectorXd &p)
  {
    assert(p.size() == loghyper.size());
    loghyper = p;
    loghyper_changed = true;
  }
  
  void CovarianceFunction::set_loghyper(const double p[])
  {
    Eigen::Map<const Eigen::VectorXd> p_vec_map(p, param_dim);
    set_loghyper(p_vec_map);
  }
  
}
