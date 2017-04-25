# libgp - Gaussian process library for Machine Learning
# Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
# All rights reserved.

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
FIND_PACKAGE(PCL REQUIRED)
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})

ADD_EXECUTABLE(test_cov test_cov.cpp)
TARGET_LINK_LIBRARIES(test_cov gp ${GPERFTOOLS_LIBRARIES})

# ADD_EXECUTABLE(gpdense gp_example_dense.cc)
# TARGET_LINK_LIBRARIES(gpdense gp ${GPERFTOOLS_LIBRARIES})
# 
ADD_EXECUTABLE(gp_vis gp_vis.cc)
TARGET_LINK_LIBRARIES(gp_vis
	${PCL_LIBRARIES}
	gp
	${GPERFTOOLS_LIBRARIES}
	)
