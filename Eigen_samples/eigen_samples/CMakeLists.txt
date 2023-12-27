cmake_minimum_required(VERSION 3.0)
project(eigen_samples)

set(EIGEN_PATH "/usr/include/eigen3" CACHE PATH "Path to Eigen")

if(NOT EIGEN_PATH)
	message(FATAL_ERROR "Missing Eigen install path, please specify EIGEN_PATH")
else()
	message("Eigen path is ${EIGEN_PATH}")

endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra ")
set(CMAKE_CXX_FLAGS_RELEASE "-03 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -0O")

include_directories(${EIGEN_PATH})

add_executable(linalg_eigen "linalg_eigen.cpp")
add_executable(linreg_eigen "linreg_eigen.cpp")
