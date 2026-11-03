cmake_minimum_required(VERSION 3.15)
project(entropic_python LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find Python
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# Fetch nanobind
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG        v2.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nanobind)

# Optional FastAndFourier for spectral/wavelet features
option(ENTROPIC_USE_FAF "Enable FastAndFourier wavelet backend" OFF)

# Build the extension module
nanobind_add_module(_entropic src/entropic_py.cpp)

# Headers: when built from repo root via scikit-build (cmake.source-dir = "python"),
# CMAKE_SOURCE_DIR is the repo root. When built standalone, try parent dir.
if(EXISTS "${CMAKE_SOURCE_DIR}/include/entropic/wfc.h")
    set(ENTROPIC_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../include/entropic/wfc.h")
    set(ENTROPIC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
else()
    message(FATAL_ERROR "Cannot find entropic headers. Build from repo root: pip install .")
endif()

target_include_directories(_entropic PRIVATE ${ENTROPIC_INCLUDE_DIR})

target_compile_definitions(_entropic PRIVATE
    WFC_ENABLE_COPULA
    WFC_ENABLE_BACKTRACK
)

if(ENTROPIC_USE_FAF)
    FetchContent_Declare(
        fastandfourier
        GIT_REPOSITORY https://github.com/NotoriousLOB/FastAndFourier.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(ENABLE_FP8 OFF CACHE BOOL "" FORCE)
    set(ENABLE_FP16 OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(fastandfourier)
    target_include_directories(fastandfourier PUBLIC
        ${fastandfourier_SOURCE_DIR}/include
        ${fastandfourier_SOURCE_DIR}/src)
    target_compile_definitions(_entropic PRIVATE WFC_USE_FAF)
    target_link_libraries(_entropic PRIVATE fastandfourier)
endif()

# Install into the Python package directory
install(TARGETS _entropic LIBRARY DESTINATION entropic)
