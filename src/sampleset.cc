// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "sampleset.h"
#include <Eigen/StdVector>

namespace libgp {

SampleSet::SampleSet (int input_dim):
	info_dim_(0)
{
    this->input_dim = input_dim;
    n = 0;
}

SampleSet::SampleSet (int input_dim, int info_dim):
	info_dim_(info_dim)
{
    this->input_dim = input_dim;
    n = 0;
}

SampleSet::SampleSet ( const SampleSet& ss )
{
    // shallow copies
    n = ss.n;
    input_dim = ss.input_dim;
	info_dim_ = ss.info_dim_;
    targets = ss.targets;

    // deep copy needed for vector of pointers
    for (size_t i=0; i<ss.inputs.size(); ++i)
    {
        Eigen::VectorXd * sample_to_store = new Eigen::VectorXd(input_dim);
        *sample_to_store = *ss.inputs.at(i);
        inputs.push_back(sample_to_store);
    }
    // deep copy needed for vector of pointers
    for (size_t i=0; i<ss.info_.size(); ++i)
    {
        Eigen::VectorXd * info_to_store = new Eigen::VectorXd(info_dim_);
        *info_to_store = *ss.info_.at(i);
        info_.push_back(info_to_store);
    }
}

SampleSet::~SampleSet()
{
    clear();
}

void SampleSet::add(const double x[], double y)
{
    Eigen::VectorXd * v = new Eigen::VectorXd(input_dim);
    for (size_t i=0; i<input_dim; ++i) (*v)(i) = x[i];
    inputs.push_back(v);
    targets.push_back(y);
    assert(inputs.size()==targets.size());
    n = inputs.size();
}

void SampleSet::add(const Eigen::VectorXd x, double y)
{
    Eigen::VectorXd * v = new Eigen::VectorXd(x);
    inputs.push_back(v);
    targets.push_back(y);
    assert(inputs.size()==targets.size());
    n = inputs.size();
}

void SampleSet::add(const double x[], double y, const double info[])
{
    Eigen::VectorXd * v = new Eigen::VectorXd(input_dim);
    for (size_t i=0; i<input_dim; ++i) (*v)(i) = x[i];
	if (info_dim_ > 0) {
		Eigen::VectorXd* s = new Eigen::VectorXd(info_dim_);
		for (size_t i = 0; i < info_dim_; ++i) (*s)(i) = info[i];
		info_.push_back(s);
	}
    inputs.push_back(v);
    targets.push_back(y);
    assert(inputs.size()==targets.size());
    n = inputs.size();
}

void SampleSet::add(const Eigen::VectorXd x, double y, const Eigen::VectorXd info)
{
    Eigen::VectorXd * v = new Eigen::VectorXd(x);
	if (info_dim_ > 0) {
		Eigen::VectorXd* s = new Eigen::VectorXd(info);
		info_.push_back(s);
	}
    inputs.push_back(v);
    targets.push_back(y);
    assert(inputs.size()==targets.size());
    n = inputs.size();
}

const Eigen::VectorXd & SampleSet::x(size_t k)
{
    return *inputs.at(k);
}

const Eigen::VectorXd & SampleSet::info (size_t k)
{
	return *info_.at(k);
}

double SampleSet::y(size_t k)
{
    return targets.at(k);
}

const std::vector<double>& SampleSet::y()
{
    return targets;
}

bool SampleSet::set_y(size_t i, double y)
{
    if (i>=n) return false;
    targets[i] = y;
    return true;
}

size_t SampleSet::size()
{
    return n;
}

void SampleSet::clear()
{
    while (!inputs.empty()) {
        delete inputs.back();
        inputs.pop_back();
    }
    n = 0;
    targets.clear();
}

bool SampleSet::empty ()
{
    return n==0;
}
}
