cmake_minimum_required(VERSION 3.0)
project(xtensor_samples)

set(XTENSOR_PATH "" CACHE PATH "Path to xtensor include dir")

if(NOT XTENSOR_PATH)
	message(FATAL_ERROR "Missing xtensor install path, please specify XTENSOR_PATH")
else()
	message("xtensor path is ${XTENSOR_PATH}")
	
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra ")
set(CMAKE_CXX_FLAGS_RELEASE "-03 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -0O")

include_directories(${XTENSOR_PATH})

add_executable(linalg_xtensor "linalg_xtensor.cpp")
target_link_libraries(linalg_xtensor cblas)
