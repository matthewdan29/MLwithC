cmake_minimum_required(VERSION 3.0)
project(eigen_recommender)

set(EIGEN_PATH "" CACHE PATH "Path to Eigen")

if(NOT EIGEN_PATH)
	message(FATAL_ERROR "Missing Eigen install path, please specify EIGEN_PATH")
else()
	message("Eigen path is ${EIGEN_PATH}")
endif()


set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra ")
set(CMAKE_CXX_FLAGS_RELEASE "-03 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -0O")

set(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")

include_directories(${EIGEN_PATH})

set(SOURCES
	eigen_recommender.cc
	data_loader.h)

add_executable(eigen_recommender ${SOURCES})
target_link_libraries(eigen_recommender ${requiredlibs})
