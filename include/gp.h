// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

/*!
 *
 *   \page licence Licensing
 *
 *     libgp - Gaussian process library for Machine Learning
 *
 *      \verbinclude "../COPYING"
 */

#ifndef __GP_H__
#define __GP_H__

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>
#include <vector>

#include "cov.h"
#include "sampleset.h"

namespace libgp {

/** Gaussian process regression.
 *  @author Manuel Blum */
class GaussianProcess
{
public:
    GaussianProcess(const GaussianProcess&) = delete;
    GaussianProcess& operator = (const GaussianProcess&) = delete;

    /** Empty initialization */
    GaussianProcess ();

    virtual ~GaussianProcess ();

    void set_cost_func(const CovarianceFunction::Ptr cf)
    {
        cost_func_ = cf;
    }

    virtual bool evaluate(struct GPData& sample, double& f, double& var);

    void add_pattern(const struct GPData& sample);

    bool set_y(size_t i, double y)
    {
        if(sampleset_.set_y(i,y)) {
            alpha_needs_update = true;
            return true;
        }
        return false;
    }
    /** Get number of samples in the training set. */
    size_t get_sampleset_size()
    {
        return sampleset_.size();
    }

    /** Clear sample set and free memory. */
    void clear_sampleset()
    {
        sampleset_.clear();
    }

    /** Get reference on currently used covariance function. */
    CovarianceFunction::Ptr covf()
    {
        return cost_func_;
    }

    /** Get input vector dimensionality. */
    size_t get_input_dim()
    {
        return sampleset_.get_input_dim();
    }

    double log_likelihood();

    Eigen::VectorXd log_likelihood_gradient();

protected:
    /** Alpha is cached for performance. */
    Eigen::VectorXd alpha;

    /** Last test kernel vector. */
    Eigen::VectorXd k_star;

    /** Linear solver used to invert the covariance matrix. */
//    Eigen::LLT<Eigen::MatrixXd> solver;
    Eigen::MatrixXd L;

    /** Update test input and cache kernel vector. */
    void update_k_star(const struct GPData &x_star);

    void update_alpha();

    /** Compute covariance matrix and perform cholesky decomposition. */
    virtual void compute();

    bool alpha_needs_update;

private:
    CovarianceFunction::Ptr cost_func_;
    SampleSet sampleset_;
};
}

#endif /* __GP_H__ */
