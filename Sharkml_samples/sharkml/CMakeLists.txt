cmake_minimum_required(VERSION 3.0)
project(sharkml-samples)

set(SHARK_PATH "/Shark/build/Shark" CACHE PATH "path to SharkML install dir")

if(NOT SHARK_PATH)
	message(FATAL_ERROR "Missing SharkML install path, please specify SHARK_PATH")
else()
	message("SharkML is ${SHARK_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED serialization)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-03 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -0O")

include_directories(${SHARK_PATH}/include)
link_directories(${SHARK_PATH}/lib)
link_directories(${SHARK_PATH}/lib64)

add_executable(linalg-shark "linalg_shark.cpp")
target_link_libraries(linalg-shark shark cblas)

add_executable(linreg-shark "linreg_shark.cpp")
target_link_libraries(linreg-shark shark cblas ${Boost_LIBRARIES} )
