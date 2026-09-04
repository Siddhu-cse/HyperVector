cmake_minimum_required(VERSION 3.16)
project(HyperVector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(FIG_ENABLE_AVX2 "Compile AVX2 SIMD path" ON)

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

add_library(hypervector STATIC
  src/hnsw_index.cpp
)
target_include_directories(hypervector PUBLIC include src)
target_compile_options(hypervector PUBLIC ${FIG_SIMD_FLAGS})

add_executable(hv_query src/hv_query.cpp)
target_link_libraries(hv_query PRIVATE hypervector)
