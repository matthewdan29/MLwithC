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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples/sharkml

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples

# Include any dependencies generated for this target.
include CMakeFiles/linreg-shark.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/linreg-shark.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/linreg-shark.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/linreg-shark.dir/flags.make

CMakeFiles/linreg-shark.dir/linreg_shark.cpp.o: CMakeFiles/linreg-shark.dir/flags.make
CMakeFiles/linreg-shark.dir/linreg_shark.cpp.o: /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples/sharkml/linreg_shark.cpp
CMakeFiles/linreg-shark.dir/linreg_shark.cpp.o: CMakeFiles/linreg-shark.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/linreg-shark.dir/linreg_shark.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/linreg-shark.dir/linreg_shark.cpp.o -MF CMakeFiles/linreg-shark.dir/linreg_shark.cpp.o.d -o CMakeFiles/linreg-shark.dir/linreg_shark.cpp.o -c /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples/sharkml/linreg_shark.cpp

CMakeFiles/linreg-shark.dir/linreg_shark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/linreg-shark.dir/linreg_shark.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples/sharkml/linreg_shark.cpp > CMakeFiles/linreg-shark.dir/linreg_shark.cpp.i

CMakeFiles/linreg-shark.dir/linreg_shark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/linreg-shark.dir/linreg_shark.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples/sharkml/linreg_shark.cpp -o CMakeFiles/linreg-shark.dir/linreg_shark.cpp.s

# Object files for target linreg-shark
linreg__shark_OBJECTS = \
"CMakeFiles/linreg-shark.dir/linreg_shark.cpp.o"

# External object files for target linreg-shark
linreg__shark_EXTERNAL_OBJECTS =

linreg-shark: CMakeFiles/linreg-shark.dir/linreg_shark.cpp.o
linreg-shark: CMakeFiles/linreg-shark.dir/build.make
linreg-shark: /usr/lib/x86_64-linux-gnu/libboost_serialization.so.1.74.0
linreg-shark: CMakeFiles/linreg-shark.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable linreg-shark"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/linreg-shark.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/linreg-shark.dir/build: linreg-shark
.PHONY : CMakeFiles/linreg-shark.dir/build

CMakeFiles/linreg-shark.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/linreg-shark.dir/cmake_clean.cmake
.PHONY : CMakeFiles/linreg-shark.dir/clean

CMakeFiles/linreg-shark.dir/depend:
	cd /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples/sharkml /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples/sharkml /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples /home/matthew/Documents/Mechine_Learning_with_C++/2Chapter/Sharkml_samples/CMakeFiles/linreg-shark.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/linreg-shark.dir/depend

