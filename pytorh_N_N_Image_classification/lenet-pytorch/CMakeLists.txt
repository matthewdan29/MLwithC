cmake_minimum_required(VERSION 3.28.1)
project(lenet-pytorch)

set(TORCH_INCLUDE_DIRS "/home/matthew/pytorch/" CACHE PATH "path to pytorch install dir")
if (NOT DEFINED TORCH_INCLUDE_DIRS)
	message(FATAL_ERROR "Missing pytorch install path, please update cmakefile")
else()
	message("Pytorch path is ${TORCH_INCLUDE_DIRS}")
endif()



set(OpenCV_LIBS "/home/matthew/Opencv/" CACHE PATH "path to opencv is installed")
if(NOT DEFINED OpenCV_LIBS)
	message(FATAL_ERROR "missing install path to Opencv")
else()
	message("OpenCV path is ${OpenCV_LIBS}")
endif()
set(CMAKE_PREFIX_PATH "/home/matthew/Documents/Opencv/")
set(CMAKE_PREFIX_PATH "/home/matthew/pytorch/")
find_package("/home/matthew/pytorch/cmake/" REQUIRED)
find_package("/home/matthew/Documents/Opencv/CMakeFiles" REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

include_directories(${TORCH_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter -pthread -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "-03 -DNDEBUG")
set(CMAKE_CXX_FLAGES_DEBUG "-ggdb -0O")

set(SOURCE_FILES main.cpp mnistdataset.h mnistdataset.cpp lenet5.h lenet5.cpp)

set(REQUIRED_LIBS "stdc++fs")
list(APPEND REQUIRED_LIBS ${TORCH_INCLUDE_DIRS})
list(APPEND REQUIRED_LIBS ${OpenCV_LIBS})

add_executable("${CMAKE_PROJECT_NAME}" ${SOURCE_FILES})
target_link_libraries("${CMAKE_PROJECT_NAME}" ${REQUIRED_LIBS})
