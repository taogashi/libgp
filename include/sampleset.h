// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __SAMPLESET_H__
#define __SAMPLESET_H__

#include <Eigen/Dense>
#include <vector>

namespace libgp {

struct GPData {
	Eigen::VectorXd x;
	Eigen::VectorXd info;
	double y;
};

/** Container holding training patterns.
 *  @author Manuel Blum */
class SampleSet
{
public:
    /** Constructor.
     *  @param input_dim dimensionality of input vectors */
    SampleSet ()
	{}

    /** Destructor. */
    virtual ~SampleSet()
	{}

	void add(const struct GPData& sample)
	{
		if (data_.empty()) {// the very first one
			input_dim_ = sample.x.size();
			info_dim_ = sample.info.size();
			data_.push_back(sample);
		} else if (sample.x.size() == input_dim_
				&& sample.info.size() == info_dim_) {
			data_.push_back(sample);
		}
	}

    /** Get input vector at index k. */
    Eigen::VectorXd x(size_t k)
	{
	 	// return wrong dimension on purpose
		if (k >= data_.size())
			return Eigen::VectorXd(input_dim_ + 1);
		return data_[k].x;
	}

    Eigen::VectorXd info(size_t k)
	{
		if (k >= data_.size())
			return Eigen::VectorXd(info_dim_ + 1);
		return data_[k].info;
	}

    /** Get target value at index k. */
    double y (size_t k)
	{
		if (k >= data_.size())
			return -1000000.0;
		return data_[k].y;
	}

	Eigen::VectorXd y()
	{
		Eigen::VectorXd y(data_.size());
		for (size_t i = 0; i < data_.size(); i++) {
			y(i) = data_[i].y;
		}
		return y;
	}

    /** Set target value at index i. */
    bool set_y(size_t i, double y)
	{
		if (i >= data_.size())
			return false;
		data_[i].y = y;
	}

    /** Get reference to vector of target values. */
    const std::vector<struct GPData>& result()
	{
		return data_;
	}

    /** Get number of samples. */
    size_t size()
	{
		return data_.size();
	}

    /** Clear sample set. */
    void clear()
	{
		data_.clear();
	}

    /** Check if sample set is empty. */
    bool empty ()
	{
		return data_.empty();
	}
	
	size_t get_input_dim()
	{
		return input_dim_;
	}

	size_t get_info_dim()
	{
		return info_dim_;
	}

private:
	std::vector<struct GPData> data_;

    /** Dimensionality of input vectors. */
    int input_dim_;
    int info_dim_;
};
}

#endif /* __SAMPLESET_H__ */
