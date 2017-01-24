# libgp - Gaussian process library for Machine Learning
# Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
# All rights reserved.

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(ENV{CMAKE_PREFIX_PATH} "/home/ys/Source/pcl/build/")
FIND_PACKAGE(PCL 1.8 REQUIRED)

INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
ADD_EXECUTABLE(gpdense gp_example_dense.cc)
TARGET_LINK_LIBRARIES(gpdense gp ${GPERFTOOLS_LIBRARIES})

ADD_EXECUTABLE(gp_vis gp_vis.cc)
TARGET_LINK_LIBRARIES(gp_vis
	${PCL_LIBRARIES}
	gp
	${GPERFTOOLS_LIBRARIES}
	)
