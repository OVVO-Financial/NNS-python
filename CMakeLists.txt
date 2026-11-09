cmake_minimum_required(VERSION 3.18)

project(nns_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

set(NNSCORE_BUILD_TESTS OFF CACHE BOOL "Build NNS-core tests" FORCE)
set(NNSCORE_BUILD_PYTHON OFF CACHE BOOL "Build NNS-core Python bindings" FORCE)
add_subdirectory(extern/NNS-core)

# Preserve the accepted NNS-core vendor tree; explicitly overlay the R14
# central-tendency kernels that its R13.3 snapshot does not yet supply, and the
# multivariate prediction kernel with R's own nmath densities.
get_target_property(NNS_CORE_SOURCES nnscore SOURCES)
set(NNS_R14_CORE_SOURCES "")
foreach(source IN LISTS NNS_CORE_SOURCES)
  if(NOT source STREQUAL "src/central_tendencies.cpp" AND NOT source STREQUAL "src/mreg_predict.cpp")
    list(APPEND NNS_R14_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/extern/NNS-core/${source}")
  endif()
endforeach()
list(APPEND NNS_R14_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/nns/_r14_central_tendencies.cpp")
list(APPEND NNS_R14_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/nns/_r14_mreg_predict.cpp")
set_property(TARGET nnscore PROPERTY SOURCES "${NNS_R14_CORE_SOURCES}")

# The Python extension is a shared object.  The vendored NNS-core target builds
# as a static library, so it must be compiled as position-independent code before
# it can be linked into the nanobind module on ELF platforms.
set_target_properties(nnscore PROPERTIES POSITION_INDEPENDENT_CODE ON)

nanobind_add_module(_nnscore src/nns/_nnscore_bindings.cpp)
target_link_libraries(_nnscore PRIVATE nnscore)
target_compile_features(_nnscore PRIVATE cxx_std_17)
install(TARGETS _nnscore LIBRARY DESTINATION nns)
