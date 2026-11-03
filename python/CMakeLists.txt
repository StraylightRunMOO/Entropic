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

# Optional FFTW for spectral features
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW QUIET fftw3)
endif()

# Build the extension module
nanobind_add_module(_entropic src/entropic_py.cpp)

target_include_directories(_entropic PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_definitions(_entropic PRIVATE
    WFC_ENABLE_COPULA
    WFC_ENABLE_BACKTRACK
)

if(FFTW_FOUND)
    target_compile_definitions(_entropic PRIVATE WFC_USE_FFTW)
    target_link_libraries(_entropic PRIVATE ${FFTW_LIBRARIES})
    target_include_directories(_entropic PRIVATE ${FFTW_INCLUDE_DIRS})
endif()

# Install into the Python package directory
install(TARGETS _entropic LIBRARY DESTINATION entropic)
