// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <ctime>

namespace libgp {

const double log2pi = log(2*M_PI);
const double initial_L_size = 1000;

GaussianProcess::GaussianProcess ()
{
	L.resize(initial_L_size, initial_L_size);
}

GaussianProcess::~GaussianProcess ()
{
}

bool GaussianProcess::evaluate(struct GPData& sample, double& f, double& var)
{
	if (sampleset_.empty()) return false;
	compute();
	update_alpha();
	update_k_star(sample);
	f = k_star.dot(alpha);
	int n = sampleset_.size();
	Eigen::VectorXd v = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(k_star);
	var = cost_func_->get(sample, sample) - v.dot(v);
	return true;
}

bool GaussianProcess::evaluate(struct GPData& sample, double& f)
{
	if (sampleset_.empty()) return false;
	compute();
	update_alpha();
	update_k_star(sample);
	f = k_star.dot(alpha);
	return true;
}

void GaussianProcess::compute()
{
    // can previously computed values be used?
    if (!cost_func_->loghyper_changed) return;
    cost_func_->loghyper_changed = false;
    int n = sampleset_.size();
    // resize L if necessary
    if (n > L.rows()) L.resize(n + initial_L_size, n + initial_L_size);
    // compute kernel matrix (lower triangle)
    for(size_t i = 0; i < sampleset_.size(); ++i) {
        for(size_t j = 0; j <= i; ++j) {
            L(i, j) = cost_func_->get(sampleset_.result()[i], sampleset_.result()[j]);
        }
    }
    // perform cholesky factorization
    //solver.compute(K.selfadjointView<Eigen::Lower>());
    L.topLeftCorner(n, n) = L.topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
    alpha_needs_update = true;
}

void GaussianProcess::update_k_star(const struct GPData &x_star)
{
    k_star.resize(sampleset_.size());
    for(size_t i = 0; i < sampleset_.size(); ++i) {
        k_star(i) = cost_func_->get(x_star, sampleset_.result()[i]);
    }
}

void GaussianProcess::update_alpha()
{
    // can previously computed values be used?
    if (!alpha_needs_update) return;
    alpha_needs_update = false;
    alpha.resize(sampleset_.size());
    // Map target values to VectorXd
	Eigen::VectorXd y = sampleset_.y();
    int n = y.size();
    alpha = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(y);
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().adjoint().solveInPlace(alpha);
}

void GaussianProcess::add_pattern(const struct GPData& sample)
{
    int n = sampleset_.size();
    sampleset_.add(sample);
    // create kernel matrix if sampleset is empty
    if (n == 0) {
        L(0,0) = sqrt(cost_func_->get(sampleset_.result()[0], sampleset_.result()[0]));
        cost_func_->loghyper_changed = false;
        // recompute kernel matrix if necessary
    } else if (cost_func_->loghyper_changed) {
        compute();
        // update kernel matrix
    } else {
        Eigen::VectorXd k(n);
        for (int i = 0; i<n; ++i) {
            k(i) = cost_func_->get(sampleset_.result()[i], sampleset_.result()[n]);
        }
        double kappa = cost_func_->get(sampleset_.result()[n], sampleset_.result()[n]);
        // resize L if necessary
        if (sampleset_.size() > static_cast<std::size_t>(L.rows())) {
            L.conservativeResize(n + initial_L_size, n + initial_L_size);
        }
        L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(k);
        L.block(n,0,1,n) = k.transpose();
        L(n,n) = sqrt(kappa - k.dot(k));
    }
    alpha_needs_update = true;
}

double GaussianProcess::log_likelihood()
{
    compute();
    update_alpha();
    int n = sampleset_.size();
	const Eigen::VectorXd y = sampleset_.y();
    double det = 2 * L.diagonal().head(n).array().log().sum();
    return -0.5*y.dot(alpha) - 0.5*det - 0.5*n*log2pi;
}

Eigen::VectorXd GaussianProcess::log_likelihood_gradient()
{
    compute();
    update_alpha();
    size_t n = sampleset_.size();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(cost_func_->get_param_dim());
    Eigen::VectorXd g(grad.size());
    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(n, n);

    // compute kernel matrix inverse
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(W);
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().transpose().solveInPlace(W);

    W = alpha * alpha.transpose() - W;

    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j <= i; ++j) {
            cost_func_->grad(sampleset_.result()[i], sampleset_.result()[j], g);
            if (i==j) grad += W(i,j) * g * 0.5;
            else      grad += W(i,j) * g;
        }
    }

    return grad;
}
}
