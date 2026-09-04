# ==============================
# CMakeLists.txt
# ==============================
cmake_minimum_required(VERSION 3.16)
project(HyperVector LANGUAGES CXX)

# ---- Standard ---------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Warnings ---------------------------------------------------------------
if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- AVX2 / FMA auto-detection ----------------------------------------------
# Enabled by default when the compiler accepts the flags. Disable with
#   cmake -DFIG_ENABLE_AVX2=OFF ..
# if you must run the binary on a CPU without AVX2 (otherwise it may SIGILL).
option(FIG_ENABLE_AVX2 "Compile the AVX2/FMA SIMD path" ON)

include(CheckCXXCompilerFlag)
set(FIG_SIMD_FLAGS "")
if(FIG_ENABLE_AVX2 AND NOT MSVC)
  check_cxx_compiler_flag("-mavx2" FIG_HAS_AVX2)
  check_cxx_compiler_flag("-mfma" FIG_HAS_FMA)
  if(FIG_HAS_AVX2)
    list(APPEND FIG_SIMD_FLAGS -mavx2)
  endif()
  if(FIG_HAS_FMA)
    list(APPEND FIG_SIMD_FLAGS -mfma)
  endif()
elseif(FIG_ENABLE_AVX2 AND MSVC)
  list(APPEND FIG_SIMD_FLAGS /arch:AVX2)
endif()

# ---- Core engine library ----------------------------------------------------
add_library(hypervector STATIC
  src/hnsw_index.cpp
)
target_include_directories(hypervector PUBLIC include src)
target_compile_options(hypervector PUBLIC ${FIG_SIMD_FLAGS})

# ---- Demo -------------------------------------------------------------------
add_executable(hv_demo examples/demo.cpp)
target_link_libraries(hv_demo PRIVATE hypervector)

add_executable(hv_query examples/hv_query.cpp)
target_link_libraries(hv_query PRIVATE hypervector)

# ---- Benchmark (note: scalar_reference.cpp is the pinned scalar baseline) ----
add_executable(hv_benchmark
  benchmarks/benchmark.cpp
  benchmarks/scalar_reference.cpp
)
target_include_directories(hv_benchmark PRIVATE benchmarks)
target_link_libraries(hv_benchmark PRIVATE hypervector)

# ---- Tests ------------------------------------------------------------------
enable_testing()
add_executable(hv_tests tests/unit_tests.cpp)
target_link_libraries(hv_tests PRIVATE hypervector)
add_test(NAME unit_tests COMMAND hv_tests)

# ---- Summary ----------------------------------------------------------------
message(STATUS "HyperVector configuration")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD flags : ${FIG_SIMD_FLAGS}")
