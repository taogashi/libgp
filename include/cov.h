// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_H__
#define __COV_H__

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <sampleset.h>

namespace libgp
{

  /** Covariance function base class.
   *  @author Manuel Blum
   *  @ingroup cov_group 
   *  @todo implement more covariance functions */
  class CovarianceFunction
  {
    public:
		typedef boost::shared_ptr<CovarianceFunction> Ptr;
		typedef boost::shared_ptr<const CovarianceFunction> ConstPtr;

      /** Constructor. */
      CovarianceFunction()
	  {};

      /** Destructor. */
      virtual ~CovarianceFunction() {};

	  virtual bool compound(Ptr cov_func)
	  {
		  return false;
	  }

      /** Computes the covariance of two input vectors.
       *  @param x1 first input vector
       *  @param x2 second input vector
       *  @return covariance of x1 and x2 */
      virtual double get(const GPData& x1, const GPData& x2) = 0;

      /** Covariance gradient of two input vectors with respect to the hyperparameters.
       *  @param x1 first input vector
       *  @param x2 second input vector
       *  @param grad covariance gradient */
      virtual void grad(const GPData& x1, const GPData& x2, Eigen::VectorXd &grad) = 0;

      /** Update parameter vector.
       *  @param p new parameter vector */
      virtual void set_loghyper(const Eigen::VectorXd &p);

      /** Update parameter vector.
       *  @param p new parameter vector */
      virtual void set_loghyper(const double p[]);

      /** Get number of parameters for this covariance function.
       *  @return parameter vector dimensionality */
      size_t get_param_dim();

      /** Get log-hyperparameter of covariance function.
       *  @return log-hyperparameter */
      Eigen::VectorXd get_loghyper();

      /** Returns a string representation of this covariance function.
       *  @return string containing the name of this covariance function */
      virtual std::string to_string() = 0;

      bool loghyper_changed;

    protected:
      /** Size of parameter vector. */
      size_t param_dim;

	  std::vector<Ptr> cov_funcs_;

      /** Parameter vector containing the log hyperparameters of the covariance function.
       *  The number of necessary parameters is given in param_dim. */
      Eigen::VectorXd loghyper;

  };

}

#endif /* __COV_H__ */

/** Covariance functions available for Gaussian process models. 
 *  There are atomic and composite covariance functions. 
 *  @defgroup cov_group Covariance Functions */
