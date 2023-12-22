cmake_minimum_required(VERSION 3.0)
project(sharkml-ensemble)

set(SHARK_PATH "/usr/local/include/shark" CACHE PATH "path to SharkML install dir")

if(NOT SHARK_PATH)
	message(FATAL_ERROR "Missing SharkML install, please specify SHARK_PAHT")
else()
	message("SharkML path is ${SHARK_PATH}")
endif()

set(PLOTCPP_PATH "/usr/share/gnuplot" CACHE PATH "path to poltcpp install dir")

if (NOT PLOTCCP_PATH)
	message(FATAL_ERROR "Missing plotcpp include path, please specify PLOPTCPP_PATH")
else()
	message("plotcpp path is ${PLOTCPP_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(Boost_USE_STATIC_LIBS ON)
SET(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost REQUIRED serialization)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-03 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -0O")

include_directories(${SHARK_PATH}/include)
include_directories(${PLOTCPP_PATH})
link_directories(${SHARK_PATH}/lib)
link_directories(${SHARK_PATH}/lib64)

set(SOURCES sharkml-ensemble.cpp)

add_executable(sharkml-ensemble ${SOURCES})
target_link_libraries(sharkml-ensemble shark cblas ${Boost_LIBRARIES} stdc++fs)
