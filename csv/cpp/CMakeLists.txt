cmake_minimum_required(VERSION 3.0)
project(csv_sample)

set(CSV_LIB_PATH "" CACHE PATH "Path to csv library include dir")
set(EIGEN_LIB_PATH "" CACHE PATH "Path to Eigen library include dir")

if (NOT CSV_LIB_PATH)
	message(FATAL_ERROR "Missing CSV lib install path, please specify CSV_LIB_PATH")
else()
	message("CSV lib path is ${CSV_PATH}")
endif()

if (NOT EIGEN_LIB_PATH)
	message(FATAL_ERROR "Missing Eigen install path, please specify EIGEN_LIB_PATH")
else()
	message("Eigen path is ${EIGEN_LIB_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra ")
set(CMAKE_CXX_FLAGS_RELEASE "-03 -DNDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG "-G -0O")

set(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")

include_directories(${CSV_LIB_PATH})
include_directories(${EIGEN_LIB_PATH})

add_executable(csv_sample "csv.cpp")
target_link_libraries(csv_sample ${requiredlibs})
