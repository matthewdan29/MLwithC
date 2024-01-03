cmake_minimum_required(VERSION 3.0)
project(sharkml-samples)

set(SHARK_PATH "/Documents/sharkml/" CACHE PATH "path to SharkML install dir")

if(NOT SHARK_PATH)
	message(FATAL_ERROR "Missing SharkML install path, please specify SHARK_PATH")
else()
	message("SharkML path is ${SHARK_PATH}")
endif()

set(PLOTCPP_PATH "/usr/lib/gnuplot/" CACHE PATH "path to poltcpp install dir ")
if (NOT PLOTCPP_PATH)
	message(FATAL_ERROR "Missing plotcpp include path, please specify PLOTCCP_PATH")
else()
	message("plotcpp path is ${PLOTCPP_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost REQUIRED serialization ON)
find_package(Boost REQUIRED boost_on-config.cmake ON)
find_package(Boost REQUIRED boost_ON_DIR ON )

set(CMAKE_CXX_FLAGS "-msse3 -fopenmp -Wall -Wextra - Wno-unnsed-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-03 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -0O")

include_directories(${SHARK_PATH}/include)
include_directories(${PLOTCPP_PATH})
link_directories(${SHARK_PATH}/lib)
link_directories(${SHARK_PATH}/lib64)

set(SOURCES polynomial-model.h
	    polynomial-regression.h
	    monitor.h
	    data.h
	    data.cpp
	    plot.h
	    plot.cpp)

add_executable(regression-shark regression-shark.cpp ${SOURCES})
target_link_libraries(regression-shark shark cblas ${Boost_LIBRARIES})

add_executable(grid-shark grid-shark.cpp)
target_link_libraries(grid-shark shark cblas ${BOOST_LIBRARIES})
