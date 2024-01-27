cmake_minimum_required(VERSION 3.10)
project(shogun-save)

set(SHOGUN_PATH "/home/matthew/Documents/Shogun/" CACHE PATH "path to shogun install dir")

if(NOT DEFINED SHOGUN_PATH)
	message("Shogun path is ${SHOGUN_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(HDF5 REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++14 -msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-03 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-G -0O")

include_directories(${HDF5_INCLUDE_DIR})
include_directories(${SHOGUN_PATH}/Include)
include_directories(/usr/include/json-c/)
include_directories(${SHOGUN_PATH}/lib)

add_executable(${CMAKE_PROJECT_NAME} "shogun-save.cpp")
target_link_libraries(${CMAKE_PROJECT_NAME} shogun)
target_link_libraries(${CMAKE_PROJECT_NAME} json-c ${HDF5_LIBRARIES} cblas lapack)
