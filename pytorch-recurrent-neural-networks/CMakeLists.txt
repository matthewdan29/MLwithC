cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(rnn-pytorch)

find_package(Torch REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-msse3 -fopenmp -Wall -Wextra -Wno-unused-paramter -pthread -fopenmp")
set(CMAKE_CXX_RELEASE "-03 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -0O")

set(SOURCE_FILES main.cpp
		 imdbataset.h
		 imdbdataset.cpp
		 rnn.h
		 rnn.cpp
		 glovedict.h
		 imdbreader.cpp
		 imdbreader.h
		 vocabulary.h
		 vocabulary.cpp
		 )

set(REQUIRED_LIBS "stdc++fs")
list(APPEND REQUIRED_LIBS ${TORCH_LIBRARIES})

add_executable("${CMAKE_PORJECT_NAME}" ${SOURCE_FILES})
target_link_libraries("${CMAKE_PROJECT_NAME}" ${REQUIRED_LIBS})
