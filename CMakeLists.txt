cmake_minimum_required(VERSION 3.14)
project(entropic VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ENTROPIC_BUILD_TESTS       "Build test suite" ON)
option(ENTROPIC_BUILD_BENCHMARKS  "Build benchmarks" ON)
option(ENTROPIC_BUILD_EXAMPLES    "Build examples" ON)
option(ENTROPIC_ENABLE_AVX2       "Enable AVX2 SIMD paths (x86_64)" OFF)
option(ENTROPIC_ENABLE_NEON       "Enable NEON SIMD paths (aarch64)" OFF)
option(ENTROPIC_ENABLE_SANITIZERS "Enable ASan/UBSan in Debug builds" OFF)
option(ENTROPIC_USE_FFTW          "Enable FFTW spectral backend" ON)
option(ENTROPIC_ENABLE_COPULA     "Enable Gaussian copula entropy biasing" ON)
option(ENTROPIC_ENABLE_BACKTRACK  "Enable delta-trail backtracking" ON)
option(ENTROPIC_BUILD_PYTHON      "Build Python bindings (requires nanobind)" OFF)

# ---------------------------------------------------------------------------
# Core library (interface header-only)
# ---------------------------------------------------------------------------
add_library(wfc INTERFACE)
add_library(wfcpp INTERFACE)
target_include_directories(wfc INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(wfcpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(wfcpp INTERFACE wfc)

# ---------------------------------------------------------------------------
# Compile definitions
# ---------------------------------------------------------------------------
if(ENTROPIC_ENABLE_AVX2)
    target_compile_definitions(wfc INTERFACE WFC_ENABLE_AVX2)
    target_compile_definitions(wfcpp INTERFACE WFC_ENABLE_AVX2)
endif()
if(ENTROPIC_ENABLE_NEON)
    target_compile_definitions(wfc INTERFACE WFC_ENABLE_NEON)
    target_compile_definitions(wfcpp INTERFACE WFC_ENABLE_NEON)
endif()
if(ENTROPIC_ENABLE_COPULA)
    target_compile_definitions(wfc INTERFACE WFC_ENABLE_COPULA)
    target_compile_definitions(wfcpp INTERFACE WFC_ENABLE_COPULA)
endif()
if(ENTROPIC_ENABLE_BACKTRACK)
    target_compile_definitions(wfc INTERFACE WFC_ENABLE_BACKTRACK)
    target_compile_definitions(wfcpp INTERFACE WFC_ENABLE_BACKTRACK)
endif()

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(wfc INTERFACE
        -Wall -Wextra -Wpedantic
        -Wno-missing-field-initializers
        $<$<CONFIG:Release>:-O3 -march=native>
        $<$<CONFIG:Debug>:-O0 -g3>
    )
    target_compile_options(wfcpp INTERFACE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Release>:-O3 -march=native>
        $<$<CONFIG:Debug>:-O0 -g3>
    )
    if(ENTROPIC_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(wfc INTERFACE -fsanitize=address,undefined)
        target_link_options(wfc INTERFACE -fsanitize=address,undefined)
        target_compile_options(wfcpp INTERFACE -fsanitize=address,undefined)
        target_link_options(wfcpp INTERFACE -fsanitize=address,undefined)
    endif()
elseif(MSVC)
    target_compile_options(wfc INTERFACE /W4)
    target_compile_options(wfcpp INTERFACE /W4)
endif()

# ---------------------------------------------------------------------------
# FetchContent for FFTW (optional) and Google Test
# ---------------------------------------------------------------------------
include(FetchContent)

if(ENTROPIC_USE_FFTW)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW QUIET fftw3)
    endif()
    if(NOT FFTW_FOUND)
        find_library(FFTW_LIB fftw3)
        find_path(FFTW_INC fftw3.h)
        if(FFTW_LIB AND FFTW_INC)
            set(FFTW_FOUND TRUE)
            set(FFTW_LIBRARIES ${FFTW_LIB})
            set(FFTW_INCLUDE_DIRS ${FFTW_INC})
        endif()
    endif()
    if(NOT FFTW_FOUND)
        message(STATUS "FFTW not found; spectral features disabled.")
        set(ENTROPIC_USE_FFTW OFF)
    else()
        target_compile_definitions(wfc INTERFACE WFC_USE_FFTW)
        target_compile_definitions(wfcpp INTERFACE WFC_USE_FFTW)
        target_link_libraries(wfc INTERFACE ${FFTW_LIBRARIES})
        target_link_libraries(wfcpp INTERFACE ${FFTW_LIBRARIES})
        target_include_directories(wfc INTERFACE ${FFTW_INCLUDE_DIRS})
        target_include_directories(wfcpp INTERFACE ${FFTW_INCLUDE_DIRS})
        message(STATUS "FFTW found: ${FFTW_LIBRARIES}")
    endif()
endif()

if(ENTROPIC_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
if(ENTROPIC_BUILD_TESTS)
    add_executable(test_wfc_c tests/test_wfc.c)
    target_link_libraries(test_wfc_c PRIVATE wfc)
    target_link_libraries(test_wfc_c PRIVATE m)
    add_test(NAME wfc_c COMMAND test_wfc_c)

    add_executable(test_wfc_gtest tests/test_wfc_basic.cpp)
    target_link_libraries(test_wfc_gtest PRIVATE wfcpp gtest_main)
    add_test(NAME wfc_gtest COMMAND test_wfc_gtest)

    add_executable(test_wfc_cpp tests/test_wfc_cpp.cpp)
    target_link_libraries(test_wfc_cpp PRIVATE wfcpp)
    add_test(NAME wfc_cpp COMMAND test_wfc_cpp)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(ENTROPIC_BUILD_BENCHMARKS)
    add_executable(bench_wfc
        benchmarks/bench_wfc.c
    )
    target_link_libraries(bench_wfc PRIVATE wfc m)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bench_wfc PRIVATE -O3)
    endif()
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(ENTROPIC_BUILD_EXAMPLES)
    add_executable(example_basic examples/example_basic.c)
    target_link_libraries(example_basic PRIVATE wfc)
    target_link_libraries(example_basic PRIVATE m)

    add_executable(example_3d examples/example_3d.c)
    target_link_libraries(example_3d PRIVATE wfc)
    target_link_libraries(example_3d PRIVATE m)

    add_executable(example_cpp examples/example_cpp.cpp)
    target_link_libraries(example_cpp PRIVATE wfcpp)

    add_executable(example_stepwise examples/example_stepwise.c)
    target_link_libraries(example_stepwise PRIVATE wfc)
    target_link_libraries(example_stepwise PRIVATE m)

    add_executable(example_cpp_advanced examples/example_cpp_advanced.cpp)
    target_link_libraries(example_cpp_advanced PRIVATE wfcpp)

    add_executable(example_kowloon examples/kowloon.c)
    target_link_libraries(example_kowloon PRIVATE wfc)
    target_link_libraries(example_kowloon PRIVATE m)

    if(ENTROPIC_USE_FFTW)
        add_executable(example_spectral examples/example_spectral.c)
        target_link_libraries(example_spectral PRIVATE wfc)
	target_link_libraries(example_spectral PRIVATE m)

        add_executable(example_copula examples/example_copula.c)
        target_link_libraries(example_copula PRIVATE wfc)
	target_link_libraries(example_copula PRIVATE m)
    endif()
endif()

# ---------------------------------------------------------------------------
# Python bindings (nanobind)
# ---------------------------------------------------------------------------
if(ENTROPIC_BUILD_PYTHON)
    find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        v2.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(nanobind)

    nanobind_add_module(_entropic python/src/entropic_py.cpp)
    target_include_directories(_entropic PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(_entropic PRIVATE
        WFC_ENABLE_COPULA
        WFC_ENABLE_BACKTRACK
    )
    if(ENTROPIC_USE_FFTW AND FFTW_FOUND)
        target_compile_definitions(_entropic PRIVATE WFC_USE_FFTW)
        target_link_libraries(_entropic PRIVATE ${FFTW_LIBRARIES})
        target_include_directories(_entropic PRIVATE ${FFTW_INCLUDE_DIRS})
    endif()
    install(TARGETS _entropic LIBRARY DESTINATION lib/python/entropic)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(FILES
    include/entropic/entropic.h
    include/entropic/wfc.h
    include/entropic/wfcpp.hpp
    include/entropic/gray_codes.h
    include/entropic/sobol.h
    DESTINATION include/entropic
)
install(TARGETS wfc wfcpp
    EXPORT entropic-targets
)
install(EXPORT entropic-targets
    FILE entropic-config.cmake
    NAMESPACE entropic::
    DESTINATION lib/cmake/entropic
)
