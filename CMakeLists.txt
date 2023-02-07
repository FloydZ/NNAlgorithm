cmake_minimum_required(VERSION 3.20)
project(nn_code CXX)

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

# only for the CI
set(BENCHMARK_ENABLE_GTEST_TESTS 0)
set(BENCHMARK_ENABLE_TESTING 0)
add_subdirectory(deps/benchmark)
#include_directories("deps/benchmark/include")
#link_directories("deps/benchmark/build/src")

include_directories("deps/b63/src")
include_directories("deps")
include_directories("src")
include_directories("test")

# two possible options. 'tcmalloc' is the standard google malloc implementation
#set(MALLOC_LIBRARY tcmalloc)
set(MALLOC_LIBRARY "")

# sadly we need '-march=native' even in 'DEBUG' mode. Because otherwise the code wouldn't compile.
# for fsized-deallocation see: https://github.com/google/tcmalloc/blob/master/docs/tuning.md
set(DEBUG_FLAGS "-g -DDEBUG -mavx2 -mavx -mbmi -mbmi2 ${VALUE_FLAG}")
set(OPTIMIZE_FLAGS "-march=native -mavx2 -mavx -mbmi -mbmi2 -O3 -funroll-loops -DNDEBUG")# -fno-inline")
set(SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer)
set(GCC_HELPER_FLAGS -fvar-tracking-assignments)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # if using GCC
    set(DEBUG_FLAGS "${DEBUG_FLAGS} ${GCC_HELPER_FLAGS} -O0 ")
endif()

# Set additionally our own debug/optimisation compiler/linkage flags. Maybe not he nicest way, but by far the easiest.
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OPTIMIZE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPTIMIZE_FLAGS}")

add_subdirectory(test)
add_subdirectory(bench)
