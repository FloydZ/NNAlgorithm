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
include bench/CMakeFiles/bench_b63_bench_instance_generator.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include bench/CMakeFiles/bench_b63_bench_instance_generator.dir/compiler_depend.make

# Include the progress variables for this target.
include bench/CMakeFiles/bench_b63_bench_instance_generator.dir/progress.make

# Include the compile flags for this target's objects.
include bench/CMakeFiles/bench_b63_bench_instance_generator.dir/flags.make

bench/CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.o: bench/CMakeFiles/bench_b63_bench_instance_generator.dir/flags.make
bench/CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.o: ../bench/bench_instance_generator.cpp
bench/CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.o: bench/CMakeFiles/bench_b63_bench_instance_generator.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object bench/CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.o"
	cd /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/bench && /nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT bench/CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.o -MF CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.o.d -o CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.o -c /home/duda/Downloads/crypto/nn/code/NNAlgorithm/bench/bench_instance_generator.cpp

bench/CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.i"
	cd /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/bench && /nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duda/Downloads/crypto/nn/code/NNAlgorithm/bench/bench_instance_generator.cpp > CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.i

bench/CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.s"
	cd /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/bench && /nix/store/lsyzzki1iv9gwk4vdss7i1cjxrnxling-gcc-wrapper-11.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duda/Downloads/crypto/nn/code/NNAlgorithm/bench/bench_instance_generator.cpp -o CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.s

# Object files for target bench_b63_bench_instance_generator
bench_b63_bench_instance_generator_OBJECTS = \
"CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.o"

# External object files for target bench_b63_bench_instance_generator
bench_b63_bench_instance_generator_EXTERNAL_OBJECTS =

bench/bench_b63_bench_instance_generator: bench/CMakeFiles/bench_b63_bench_instance_generator.dir/bench_instance_generator.cpp.o
bench/bench_b63_bench_instance_generator: bench/CMakeFiles/bench_b63_bench_instance_generator.dir/build.make
bench/bench_b63_bench_instance_generator: bench/CMakeFiles/bench_b63_bench_instance_generator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bench_b63_bench_instance_generator"
	cd /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/bench && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bench_b63_bench_instance_generator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bench/CMakeFiles/bench_b63_bench_instance_generator.dir/build: bench/bench_b63_bench_instance_generator
.PHONY : bench/CMakeFiles/bench_b63_bench_instance_generator.dir/build

bench/CMakeFiles/bench_b63_bench_instance_generator.dir/clean:
	cd /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/bench && $(CMAKE_COMMAND) -P CMakeFiles/bench_b63_bench_instance_generator.dir/cmake_clean.cmake
.PHONY : bench/CMakeFiles/bench_b63_bench_instance_generator.dir/clean

bench/CMakeFiles/bench_b63_bench_instance_generator.dir/depend:
	cd /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/duda/Downloads/crypto/nn/code/NNAlgorithm /home/duda/Downloads/crypto/nn/code/NNAlgorithm/bench /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/bench /home/duda/Downloads/crypto/nn/code/NNAlgorithm/build/bench/CMakeFiles/bench_b63_bench_instance_generator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bench/CMakeFiles/bench_b63_bench_instance_generator.dir/depend

