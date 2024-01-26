cmake_minimum_required(VERSION 3.10)
project(shogun-save)

set(SHOGUN_PATH "/home/Documents/shogun/" CACHE PATH "path to shogun install dir")

if(NOT DEFINED SHOGUN_PATH)
	message(FATAL_ERROR "Missing shogun install path, please specify SHOGUN_PATH")
else()
	message("shogun path is ${SHOGUN_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(HDF5 REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++14 -msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-03 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -0O")

include_directories(${HDF5_INCLUDE_DIR})
include_directories(${SHOGUN_PATH}/include)
include_directories(/usr/include/json-c/)
link_directories(${SHOGUN_PATH}/lib)

add_executable(${CMAKE_PORJECT_NAME} "shogun-save.cpp")
target_link_libraries(${CMAKE_PROJECT_NAME} shogun)
target_link_libraries(${CMAKE_PROJECT_NAME} json- ${HDF5_LIBRARIES} cblas lapack)


