# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification

# Include any dependencies generated for this target.
include CMakeFiles/lenet-pytorch.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/lenet-pytorch.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/lenet-pytorch.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lenet-pytorch.dir/flags.make

CMakeFiles/lenet-pytorch.dir/main.cpp.o: CMakeFiles/lenet-pytorch.dir/flags.make
CMakeFiles/lenet-pytorch.dir/main.cpp.o: /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch/main.cpp
CMakeFiles/lenet-pytorch.dir/main.cpp.o: CMakeFiles/lenet-pytorch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lenet-pytorch.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/lenet-pytorch.dir/main.cpp.o -MF CMakeFiles/lenet-pytorch.dir/main.cpp.o.d -o CMakeFiles/lenet-pytorch.dir/main.cpp.o -c /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch/main.cpp

CMakeFiles/lenet-pytorch.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/lenet-pytorch.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch/main.cpp > CMakeFiles/lenet-pytorch.dir/main.cpp.i

CMakeFiles/lenet-pytorch.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/lenet-pytorch.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch/main.cpp -o CMakeFiles/lenet-pytorch.dir/main.cpp.s

CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.o: CMakeFiles/lenet-pytorch.dir/flags.make
CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.o: /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch/mnistdataset.cpp
CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.o: CMakeFiles/lenet-pytorch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.o -MF CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.o.d -o CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.o -c /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch/mnistdataset.cpp

CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch/mnistdataset.cpp > CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.i

CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch/mnistdataset.cpp -o CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.s

CMakeFiles/lenet-pytorch.dir/lenet5.cpp.o: CMakeFiles/lenet-pytorch.dir/flags.make
CMakeFiles/lenet-pytorch.dir/lenet5.cpp.o: /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch/lenet5.cpp
CMakeFiles/lenet-pytorch.dir/lenet5.cpp.o: CMakeFiles/lenet-pytorch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/lenet-pytorch.dir/lenet5.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/lenet-pytorch.dir/lenet5.cpp.o -MF CMakeFiles/lenet-pytorch.dir/lenet5.cpp.o.d -o CMakeFiles/lenet-pytorch.dir/lenet5.cpp.o -c /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch/lenet5.cpp

CMakeFiles/lenet-pytorch.dir/lenet5.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/lenet-pytorch.dir/lenet5.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch/lenet5.cpp > CMakeFiles/lenet-pytorch.dir/lenet5.cpp.i

CMakeFiles/lenet-pytorch.dir/lenet5.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/lenet-pytorch.dir/lenet5.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch/lenet5.cpp -o CMakeFiles/lenet-pytorch.dir/lenet5.cpp.s

# Object files for target lenet-pytorch
lenet__pytorch_OBJECTS = \
"CMakeFiles/lenet-pytorch.dir/main.cpp.o" \
"CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.o" \
"CMakeFiles/lenet-pytorch.dir/lenet5.cpp.o"

# External object files for target lenet-pytorch
lenet__pytorch_EXTERNAL_OBJECTS =

/home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch: CMakeFiles/lenet-pytorch.dir/main.cpp.o
/home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch: CMakeFiles/lenet-pytorch.dir/mnistdataset.cpp.o
/home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch: CMakeFiles/lenet-pytorch.dir/lenet5.cpp.o
/home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch: CMakeFiles/lenet-pytorch.dir/build.make
/home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch: /home/matthew/Opencv
/home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch: CMakeFiles/lenet-pytorch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lenet-pytorch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lenet-pytorch.dir/build: /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch
.PHONY : CMakeFiles/lenet-pytorch.dir/build

CMakeFiles/lenet-pytorch.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lenet-pytorch.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lenet-pytorch.dir/clean

CMakeFiles/lenet-pytorch.dir/depend:
	cd /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/lenet-pytorch /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification /home/matthew/Documents/Mechine_Learning_with_C++/11Chapter_neural_networks_for_image_classification/pytorh_N_N_Image_classification/CMakeFiles/lenet-pytorch.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/lenet-pytorch.dir/depend
