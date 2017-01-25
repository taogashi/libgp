// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_sum.h"
#include "cmath"

namespace libgp
{

CovSum::CovSum()
{
	param_dim = 0;
}

CovSum::~CovSum()
{ }

bool CovSum::compound(Ptr cov_func)
{
    if (!cov_func) {
        return false;
    }
    param_dim += cov_func->get_param_dim();
    cov_funcs_.push_back(cov_func);
	Eigen::VectorXd tmp(loghyper.size() + cov_func->get_param_dim());
	tmp << loghyper, cov_func->get_loghyper();
	loghyper = tmp;
    return true;
}

double CovSum::get(const GPData& x1, const GPData& x2)
{
    double cov = 0.0;
    for (auto func : cov_funcs_) {
        cov += func->get(x1, x2);
    }
    return cov;
}

void CovSum::grad(const GPData& x1, const GPData& x2, Eigen::VectorXd &grad)
{
    grad.resize(param_dim);
    int start_idx = 0;
    for (auto func : cov_funcs_) {
        Eigen::VectorXd grad_seg(func->get_param_dim());
        func->grad(x1, x2, grad_seg);
        grad.segment(start_idx, start_idx + func->get_param_dim() - 1)
            = grad_seg;
        start_idx += func->get_param_dim();
    }
}

void CovSum::set_loghyper(const Eigen::VectorXd &p)
{
    CovarianceFunction::set_loghyper(p);
    int start_idx = 0;
    for (auto func : cov_funcs_) {
        func->set_loghyper(p.segment(start_idx, start_idx + func->get_param_dim()));
        start_idx += func->get_param_dim();
    }
}

std::string CovSum::to_string()
{
    return "CovSum";
}
}
