# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /nix/store/k7lm30wld0jhdks4maz47v7ak8ydv2g6-cmake-3.22.3/bin/cmake

# The command to remove a file.
RM = /nix/store/k7lm30wld0jhdks4maz47v7ak8ydv2g6-cmake-3.22.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/duda/Downloads/crypto/nn/code/NNAlgorithm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build

# Include any dependencies generated for this target.
include test/CMakeFiles/test_nn_golden.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/test_nn_golden.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_nn_golden.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_nn_golden.dir/flags.make

test/CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.o: test/CMakeFiles/test_nn_golden.dir/flags.make
test/CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.o: ../test/test_nn_golden.cpp
test/CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.o: test/CMakeFiles/test_nn_golden.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.o"
	cd /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/test && /nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.o -MF CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.o.d -o CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.o -c /home/duda/Downloads/crypto/nn/code/NNAlgorithm/test/test_nn_golden.cpp

test/CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.i"
	cd /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/test && /nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duda/Downloads/crypto/nn/code/NNAlgorithm/test/test_nn_golden.cpp > CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.i

test/CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.s"
	cd /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/test && /nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duda/Downloads/crypto/nn/code/NNAlgorithm/test/test_nn_golden.cpp -o CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.s

# Object files for target test_nn_golden
test_nn_golden_OBJECTS = \
"CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.o"

# External object files for target test_nn_golden
test_nn_golden_EXTERNAL_OBJECTS =

test/test_nn_golden: test/CMakeFiles/test_nn_golden.dir/test_nn_golden.cpp.o
test/test_nn_golden: test/CMakeFiles/test_nn_golden.dir/build.make
test/test_nn_golden: test/CMakeFiles/test_nn_golden.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_nn_golden"
	cd /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_nn_golden.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_nn_golden.dir/build: test/test_nn_golden
.PHONY : test/CMakeFiles/test_nn_golden.dir/build

test/CMakeFiles/test_nn_golden.dir/clean:
	cd /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/test && $(CMAKE_COMMAND) -P CMakeFiles/test_nn_golden.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_nn_golden.dir/clean

test/CMakeFiles/test_nn_golden.dir/depend:
	cd /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/duda/Downloads/crypto/nn/code/NNAlgorithm /home/duda/Downloads/crypto/nn/code/NNAlgorithm/test /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/test /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/test/CMakeFiles/test_nn_golden.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_nn_golden.dir/depend

