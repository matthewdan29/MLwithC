cmake_minimum_required(VERSION 3.10)
project(shogun-csv-sample)

set(SHOGUN_PATH "" CACHE PATH "path to shogun install dir")

if(NOT DEFINED SHOGUN_PATH)
	message(FATAL_ERROR "Missing Shogun install path, please specify SHOGUN_PATH")
else()
	message("Shogun path is ${SHOGUN_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++14 -msse3 -fopenmp -Wall -Wextra -Wno-unsuded-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-03 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -0O")

set(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")
list(APPEND requiredlibs json-c ${PROTOBUF_LIBRARIES} cblas lapack)

include_directories(${SHOGUN_PATH}/include)
link_directories(${SHOGUN_PATH}/lib)

add_executable(csv_shogun "csv_shogun.cpp")
target_link_libraries(csv_shogun shogun ${requriedlibs})
