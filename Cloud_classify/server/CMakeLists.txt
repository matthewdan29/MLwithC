cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(classify-server)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)

find_package(TORCH REQUIRED)
find_package(OpenCV REQUIRED)

include_directories("third-party/")
include_directories(${CMAKE_PREFIX_PATH}/include)
include_directories(${ONNX_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter -pthread -fopenmp -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-03 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -0O")

set(SOURCE_FILES main.cpp 
		utils.h
		utils.cpp 
		network.h
		network.cpp)

set(REQUIRED_LIBS "")
list(APPEND REQUIRED_LIBS ${TORCH_LIBRARIES})
list(APPEND REQUIRED_LIBS ${OpenCV_LIBRARIES})

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${REQUIRED_LIBS})
