cmake_minimum_required(VERSION 3.10)
project(dlib-save)

set(DLIB_PATH "/home/matthew/Documents/Dlib" CACHE PATH "path to dlib install dir")

if (NOT DLIB_PATH)
	message(FATAL_ERROR "Missing Dlib install path, please specify DLIB_PATH")
else()
	message("Dlib path is ${DLIB_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(requiredlibs "cblas")
list(APPEND requiredlibs "lapack")

find_package(CUDA)
if (CUDA)
	list(APPEND requiredlibs ${CUDA_LIBRARIES})
	list(APPEND requiredlibs ${CUDA_CUBLAS_LIBRARIES})
	list(APPEND requiredlibs ${CUDA_cusolver_LIBRARY})
	list(APPEND requiredlibs ${CUDA_curand_LIBRARY})

	get_filename_component(cudnn_hint_path "${CUDA_CUBLAS_LIBRARIES}" PATH)
	find_library(cudnn cudnn
		HINTS ${cudnn_hint_path} ENV CUDNN_LIBRRAY_DIR ENV CUDNN_HOME
		PATHS /usr/local /usr/local/cuda ENV LD_LIBRARY_PATH
		PATH_SUFFIXES lib64 lib x64
		)
	message("CUDNN found ${cudnn}")
	list(APPEND requiredlibs ${cudnn})
endif()

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-03 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -0O")

include_directories(${DLIB_PATH}/include)
link_directories(${DLIB_PATH}/lib)
link_directories(${DLIB_PATH}/lib64)

add_executable(${CMAKE_PROJECT_NAME} "dlib-save.cpp")
target_link_libraries(${CMAKE_PROJECT_NAME} optimized dlib debug dlibd)
target_link_libraries(${CMAKE_PROJECT_NAME} ${requiredlibs})
