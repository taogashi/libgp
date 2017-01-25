// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_SUM_H__
#define __COV_SUM_H__

#include "cov.h"

namespace libgp
{
/** Sums of covariance functions.
 *  @author Manuel Blum
 *  @ingroup cov_group */
class CovSum : public CovarianceFunction
{
public:
    CovSum ();
    virtual ~CovSum ();
    bool compound(Ptr cov_func);
    double get(const GPData& x1, const GPData& x2);
    void grad(const GPData& x1, const GPData& x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
private:
};

}

#endif /* __COV_SUM_H__ */
