# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples/eigen_samples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples

# Include any dependencies generated for this target.
include CMakeFiles/linalg_eigen.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/linalg_eigen.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/linalg_eigen.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/linalg_eigen.dir/flags.make

CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.o: CMakeFiles/linalg_eigen.dir/flags.make
CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.o: /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples/eigen_samples/linalg_eigen.cpp
CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.o: CMakeFiles/linalg_eigen.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.o -MF CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.o.d -o CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.o -c /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples/eigen_samples/linalg_eigen.cpp

CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples/eigen_samples/linalg_eigen.cpp > CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.i

CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples/eigen_samples/linalg_eigen.cpp -o CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.s

# Object files for target linalg_eigen
linalg_eigen_OBJECTS = \
"CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.o"

# External object files for target linalg_eigen
linalg_eigen_EXTERNAL_OBJECTS =

linalg_eigen: CMakeFiles/linalg_eigen.dir/linalg_eigen.cpp.o
linalg_eigen: CMakeFiles/linalg_eigen.dir/build.make
linalg_eigen: CMakeFiles/linalg_eigen.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable linalg_eigen"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/linalg_eigen.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/linalg_eigen.dir/build: linalg_eigen
.PHONY : CMakeFiles/linalg_eigen.dir/build

CMakeFiles/linalg_eigen.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/linalg_eigen.dir/cmake_clean.cmake
.PHONY : CMakeFiles/linalg_eigen.dir/clean

CMakeFiles/linalg_eigen.dir/depend:
	cd /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples/eigen_samples /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples/eigen_samples /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Eigen_samples/CMakeFiles/linalg_eigen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/linalg_eigen.dir/depend

