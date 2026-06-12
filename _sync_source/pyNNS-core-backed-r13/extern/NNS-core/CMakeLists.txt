cmake_minimum_required(VERSION 3.18)

# Project name and version lockstep with NNS releases
project(nnscore VERSION 13.0.0 LANGUAGES CXX)

# Enforce C++17 Standard (Required for std::optional and modern standard library features)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define the core library and its source files
add_library(nnscore
    src/partial_moments.cpp
    src/central_tendencies.cpp
    src/partition.cpp
    src/distance.cpp
    src/stochastic_dominance.cpp
    src/dependence.cpp
    src/seasonality.cpp
    src/fast_lm.cpp
    src/internal_functions.cpp
)

# Set the public include directories so consumers can `#include "nns/nns.hpp"`
target_include_directories(nnscore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link the system threading library (Replaces RcppParallel backend)
find_package(Threads REQUIRED)
target_link_libraries(nnscore PUBLIC Threads::Threads)

# ---------------------------------------------------------
# Build Options & Subdirectories
# ---------------------------------------------------------

option(NNSCORE_BUILD_TESTS "Build Catch2 unit and conformance tests" ON)
option(NNSCORE_BUILD_PYTHON "Build Python bindings via nanobind" OFF)

if(NNSCORE_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

if(NNSCORE_BUILD_PYTHON)
    add_subdirectory(bindings/python)
endif()