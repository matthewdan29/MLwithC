# For more about information about using CMake with android Studio, read 
# documentation: https://d.android.com/studio/projects/add-native-code.html

#Sets the minimum version of Cmake required to build the native library. 

cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(PATH_TO_PYTORCH CACHE STRING "/home/matthew/Pytorch/")

set(THREADS_PREFER_PTHREADS_FLAG ON)
find_package(Threads REQUIRED)

include_directories(${PATH_TO_PYTORCH}/include)

add_definitions(-DONNX_NAMESPACE=onnx_torch)

#Creates and names a library, sets it as either STATIC or SHARED, and provides the relative paths to its source code. 
#you can define multiple libraries, and CMake builds them for you. 
#Gradle automatically packages shared libraries with your APK

add_library(#sets the name of the library
		native-lib
		
		# Sets the library as a shared library
		SHARED 
		
		#Provides a relative path to your soucre files(s)
		native-lib.cpp)

# Searches for a specified prebuilt library and stores the path as a variable. 
#Because CMake includes system libraries in the search path by default, you only need to specify the name ofthe public NDK library you want to add.
# Cmake verifies that library exists before completing its build.

find_library(#Sets the name of the path veriable.
		log-lib
		
		#specifies the name of the NDK library that you want CMake to loacte 
		log)

#Specifies libraries CMake should link to your target library. 
#You can link multiple libraries, such as libraries you define in this build script, prebuilt third-party libraries,(that take hours to build and when you turn off your computer its still processing like my guys this is pushing this computer to the max) or system libraries. 

set(pytorch-lib ${PATH_TO_PYTORCH}/lib/libtorch.so)
set(c10-lib ${PATH_TO_TORCH}/lib/libc10.so)

#set(pytorch-lib ${PATH_TO_TORCH}/lib/libtorch.a)
#set(c10-lib ${PATH_TO_PYTORCH}/lib/libc10.a)
#set(cpuinfo-lib ${PATH_TO_PYTORCH}/lib/libcpuinfo.a)
#set(clog-lib ${PATH_TO_PYTORCH}/lib/libclog.a)
#set(caffe2-protos-lib ${PATH_TO_PYTORCH}/lib/libcaffe2_protos.a)
#set(protobuf-lib ${PATH_TO_PYTORCH}/lib/libprotobuf.a)
#set(nnpack-lib ${PATH_TO_PYTORCH}/lib/libnnpack.a)
#set(qnnpack-lib ${PATH_TO_PYTORCH}/lib/libqnnpack.a)
#set(thread-lib ${PATH_TO_PYTORCH}/lib/libpthreadpool.a)

target_link_libraries(#specifies the target library
	native-lib
	
	# -Wl, --whole-archive -Wl, --no-as-needed
	${log-lib}
	${pytorch-lib}
	${c1--lib}
	# ${cpuinfo-lib} ...lol if you know you know
	# ${clog-lib}
	# ${caffe2-protos-lib}
	# ${protobuf-lib}
	# ${nnpack-lib}
	# ${qnnpack-lib}
	# ${thread-lib}
	android
	)
