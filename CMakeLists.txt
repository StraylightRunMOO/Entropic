cmake_minimum_required(VERSION 3.14)
project(entropic VERSION 1.1.0 LANGUAGES C CXX)

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
option(ENTROPIC_USE_FAF            "Enable FastAndFourier wavelet backend" OFF)
option(ENTROPIC_ENABLE_HIERARCHY   "Enable multiscale hierarchy (requires FAF)" OFF)
option(ENTROPIC_ENABLE_SUSPENDERS  "Enable suspenders.h coroutine parallelism (requires FAF)" OFF)
option(ENTROPIC_ENABLE_COPULA     "Enable Gaussian copula entropy biasing" ON)
option(ENTROPIC_ENABLE_BACKTRACK  "Enable delta-trail backtracking" ON)
option(ENTROPIC_BUILD_PYTHON      "Build Python bindings (requires nanobind)" OFF)
option(ENTROPIC_ENABLE_FUZZ       "Build fuzz targets (libFuzzer + standalone)" OFF)
option(ENTROPIC_ENABLE_COVERAGE   "Instrument tests for llvm-cov / gcov" OFF)

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
# FetchContent dependencies
# ---------------------------------------------------------------------------
include(FetchContent)

if(ENTROPIC_USE_FAF)
    FetchContent_Declare(
        fastandfourier
        GIT_REPOSITORY https://github.com/NotoriousLOB/FastAndFourier.git
        GIT_TAG        f53137674929b86b8d50aef949a926920248c50e  # pin for v1.1.0
        GIT_SHALLOW    FALSE
    )
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(ENABLE_FP8 OFF CACHE BOOL "" FORCE)
    set(ENABLE_FP16 OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(fastandfourier)
    # FAF's CMakeLists uses CMAKE_SOURCE_DIR (wrong when consumed via
    # FetchContent).  Patch include dirs on the target directly.
    target_include_directories(fastandfourier PUBLIC
        ${fastandfourier_SOURCE_DIR}/include
        ${fastandfourier_SOURCE_DIR}/src)
    target_compile_definitions(wfc INTERFACE WFC_USE_FAF)
    target_compile_definitions(wfcpp INTERFACE WFC_USE_FAF)
    target_link_libraries(wfc INTERFACE $<BUILD_INTERFACE:fastandfourier>)
    target_link_libraries(wfcpp INTERFACE $<BUILD_INTERFACE:fastandfourier>)
    target_include_directories(wfc INTERFACE
        $<BUILD_INTERFACE:${fastandfourier_SOURCE_DIR}/include>)
    target_include_directories(wfcpp INTERFACE
        $<BUILD_INTERFACE:${fastandfourier_SOURCE_DIR}/include>)
    if(ENTROPIC_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(fastandfourier PRIVATE -fsanitize=address,undefined)
        target_link_options(fastandfourier PUBLIC -fsanitize=address,undefined)
    endif()
    message(STATUS "FastAndFourier enabled (wavelet backend; JIT uses cc at runtime, VM fallback otherwise)")
endif()

if(ENTROPIC_ENABLE_HIERARCHY)
    if(NOT ENTROPIC_USE_FAF)
        message(FATAL_ERROR "ENTROPIC_ENABLE_HIERARCHY requires ENTROPIC_USE_FAF")
    endif()
    target_compile_definitions(wfc INTERFACE WFC_ENABLE_HIERARCHY)
    target_compile_definitions(wfcpp INTERFACE WFC_ENABLE_HIERARCHY)
endif()

if(ENTROPIC_ENABLE_SUSPENDERS)
    if(NOT ENTROPIC_USE_FAF)
        message(FATAL_ERROR "ENTROPIC_ENABLE_SUSPENDERS requires ENTROPIC_USE_FAF")
    endif()

    # Memento (header-only allocator, required by suspenders)
    FetchContent_Declare(
        memento
        GIT_REPOSITORY https://github.com/StraylightRunMOO/memento.git
        GIT_TAG        v3.0.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_Populate(memento)

    # Suspenders (header-only coroutines)
    FetchContent_Declare(
        suspenders
        GIT_REPOSITORY https://github.com/StraylightRunMOO/suspenders.h.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    FetchContent_Populate(suspenders)

    target_compile_definitions(wfc INTERFACE WFC_ENABLE_SUSPENDERS)
    target_compile_definitions(wfcpp INTERFACE WFC_ENABLE_SUSPENDERS)
    target_include_directories(wfc INTERFACE
        $<BUILD_INTERFACE:${suspenders_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${memento_SOURCE_DIR}/include>)
    target_include_directories(wfcpp INTERFACE
        $<BUILD_INTERFACE:${suspenders_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${memento_SOURCE_DIR}/include>)

    # Suspenders needs pthreads
    find_package(Threads REQUIRED)
    target_link_libraries(wfc INTERFACE Threads::Threads)
    target_link_libraries(wfcpp INTERFACE Threads::Threads)

    # Auto-detect liburing on Linux; fall back to poll backend if absent
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(LIBURING QUIET liburing>=2.0)
        endif()
        if(LIBURING_FOUND)
            target_include_directories(wfc INTERFACE ${LIBURING_INCLUDE_DIRS})
            target_link_libraries(wfc INTERFACE ${LIBURING_LIBRARIES})
            target_include_directories(wfcpp INTERFACE ${LIBURING_INCLUDE_DIRS})
            target_link_libraries(wfcpp INTERFACE ${LIBURING_LIBRARIES})
            message(STATUS "Suspenders: using io_uring backend")
        else()
            target_compile_definitions(wfc INTERFACE SUSPENDERS_FORCE_POLL)
            target_compile_definitions(wfcpp INTERFACE SUSPENDERS_FORCE_POLL)
            message(STATUS "Suspenders: liburing not found, using poll backend")
        endif()
    endif()

    message(STATUS "Suspenders.h coroutine parallelism enabled")
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
# Coverage helper
# ---------------------------------------------------------------------------
function(entropic_apply_coverage tgt)
    if(NOT ENTROPIC_ENABLE_COVERAGE)
        return()
    endif()
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        target_compile_options(${tgt} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(${tgt} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${tgt} PRIVATE --coverage)
        target_link_options(${tgt} PRIVATE --coverage)
    endif()
endfunction()

# Keep assert() live in tests even when the rest of the tree is Release.
function(entropic_test_defaults tgt)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${tgt} PRIVATE -UNDEBUG)
    elseif(MSVC)
        target_compile_options(${tgt} PRIVATE /UNDEBUG)
    endif()
    entropic_apply_coverage(${tgt})
endfunction()

# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
if(ENTROPIC_BUILD_TESTS)
    add_executable(test_wfc_c tests/test_wfc.c)
    target_link_libraries(test_wfc_c PRIVATE wfc)
    target_link_libraries(test_wfc_c PRIVATE m)
    entropic_test_defaults(test_wfc_c)
    add_test(NAME wfc_c COMMAND test_wfc_c)

    add_executable(test_wfc_gtest tests/test_wfc_basic.cpp)
    target_link_libraries(test_wfc_gtest PRIVATE wfcpp gtest_main)
    entropic_test_defaults(test_wfc_gtest)
    add_test(NAME wfc_gtest COMMAND test_wfc_gtest)

    add_executable(test_wfc_cpp tests/test_wfc_cpp.cpp)
    target_link_libraries(test_wfc_cpp PRIVATE wfcpp)
    entropic_test_defaults(test_wfc_cpp)
    add_test(NAME wfc_cpp COMMAND test_wfc_cpp)

    add_executable(test_gray_sobol tests/test_gray_sobol.c)
    target_link_libraries(test_gray_sobol PRIVATE wfc m)
    entropic_test_defaults(test_gray_sobol)
    add_test(NAME gray_sobol COMMAND test_gray_sobol)

    add_executable(test_quality tests/test_quality.c)
    target_link_libraries(test_quality PRIVATE wfc m)
    entropic_test_defaults(test_quality)
    add_test(NAME quality COMMAND test_quality)

    if(ENTROPIC_USE_FAF)
        add_executable(test_faf_smoke tests/test_faf_dwt.c)
        target_link_libraries(test_faf_smoke PRIVATE fastandfourier m)
        target_include_directories(test_faf_smoke PRIVATE
            ${fastandfourier_SOURCE_DIR}/include)
        entropic_test_defaults(test_faf_smoke)
        add_test(NAME faf_smoke COMMAND test_faf_smoke)
    endif()

    if(ENTROPIC_ENABLE_HIERARCHY)
        add_executable(test_hierarchy tests/test_hierarchy.c)
        target_link_libraries(test_hierarchy PRIVATE wfc m)
        entropic_test_defaults(test_hierarchy)
        add_test(NAME hierarchy COMMAND test_hierarchy)
    endif()

    if(ENTROPIC_ENABLE_SUSPENDERS)
        add_executable(test_suspenders tests/test_suspenders.c)
        target_link_libraries(test_suspenders PRIVATE wfc m)
        entropic_test_defaults(test_suspenders)
        add_test(NAME suspenders COMMAND test_suspenders)
    endif()
endif()

# ---------------------------------------------------------------------------
# Fuzz testing
# ---------------------------------------------------------------------------
if(ENTROPIC_ENABLE_FUZZ)
    add_executable(fuzz_wfc_standalone tests/fuzz_wfc.c)
    target_link_libraries(fuzz_wfc_standalone PRIVATE wfc m)
    target_compile_definitions(fuzz_wfc_standalone PRIVATE FUZZ_STANDALONE)
    entropic_test_defaults(fuzz_wfc_standalone)
    if(ENTROPIC_BUILD_TESTS)
        add_test(NAME fuzz_standalone COMMAND fuzz_wfc_standalone)
        set_tests_properties(fuzz_standalone PROPERTIES
            ENVIRONMENT "ENTROPIC_FUZZ_ITERS=400")
    endif()

    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        add_executable(fuzz_wfc tests/fuzz_wfc.c)
        target_link_libraries(fuzz_wfc PRIVATE wfc m)
        target_compile_options(fuzz_wfc PRIVATE -fsanitize=fuzzer,address -fno-omit-frame-pointer)
        target_link_options(fuzz_wfc PRIVATE -fsanitize=fuzzer,address)
    endif()
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

    add_executable(example_learn examples/example_learn.c)
    target_link_libraries(example_learn PRIVATE wfc m)

    add_executable(example_backtrack examples/example_backtrack.c)
    target_link_libraries(example_backtrack PRIVATE wfc m)

    add_executable(example_seamless examples/example_seamless.c)
    target_link_libraries(example_seamless PRIVATE wfc m)

    add_executable(example_gray_sobol examples/example_gray_sobol.c)
    target_link_libraries(example_gray_sobol PRIVATE wfc m)

    add_executable(example_cpp_3d examples/example_cpp_3d.cpp)
    target_link_libraries(example_cpp_3d PRIVATE wfcpp)

    add_executable(example_cpp_tour examples/example_cpp_tour.cpp)
    target_link_libraries(example_cpp_tour PRIVATE wfcpp)

    if(ENTROPIC_USE_FAF)
        add_executable(example_spectral examples/example_spectral.c)
        target_link_libraries(example_spectral PRIVATE wfc m)

        add_executable(example_cpp_wavelets examples/example_cpp_wavelets.cpp)
        target_link_libraries(example_cpp_wavelets PRIVATE wfcpp)
    endif()

    add_executable(example_copula examples/example_copula.c)
    target_link_libraries(example_copula PRIVATE wfc m)

    if(ENTROPIC_ENABLE_HIERARCHY)
        add_executable(example_hierarchy examples/example_hierarchy.c)
        target_link_libraries(example_hierarchy PRIVATE wfc m)
    endif()

    if(ENTROPIC_ENABLE_SUSPENDERS)
        add_executable(example_parallel examples/example_parallel.c)
        target_link_libraries(example_parallel PRIVATE wfc m)
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
    if(ENTROPIC_USE_FAF)
        target_compile_definitions(_entropic PRIVATE WFC_USE_FAF)
        target_link_libraries(_entropic PRIVATE fastandfourier)
    endif()
    if(ENTROPIC_ENABLE_HIERARCHY)
        target_compile_definitions(_entropic PRIVATE WFC_ENABLE_HIERARCHY)
    endif()
    if(ENTROPIC_ENABLE_SUSPENDERS)
        target_compile_definitions(_entropic PRIVATE WFC_ENABLE_SUSPENDERS)
        target_include_directories(_entropic PRIVATE
            ${suspenders_SOURCE_DIR}/include
            ${memento_SOURCE_DIR}/include)
        target_link_libraries(_entropic PRIVATE Threads::Threads)
        if(LIBURING_FOUND)
            target_include_directories(_entropic PRIVATE ${LIBURING_INCLUDE_DIRS})
            target_link_libraries(_entropic PRIVATE ${LIBURING_LIBRARIES})
        else()
            target_compile_definitions(_entropic PRIVATE SUSPENDERS_FORCE_POLL)
        endif()
    endif()
    entropic_apply_coverage(_entropic)
    install(TARGETS _entropic LIBRARY DESTINATION lib/python/entropic)

    # Stage package layout for in-tree pytest / python benchmarks
    set(ENTROPIC_PY_STAGE "${CMAKE_BINARY_DIR}/python")
    add_custom_command(TARGET _entropic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${ENTROPIC_PY_STAGE}/entropic"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/python/entropic/__init__.py"
            "${ENTROPIC_PY_STAGE}/entropic/__init__.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:_entropic>"
            "${ENTROPIC_PY_STAGE}/entropic/$<TARGET_FILE_NAME:_entropic>"
        COMMENT "Staging Python package under ${ENTROPIC_PY_STAGE}"
    )
    if(ENTROPIC_BUILD_TESTS)
        add_test(NAME python_pytest
            COMMAND ${Python_EXECUTABLE} -m pytest
                "${CMAKE_CURRENT_SOURCE_DIR}/python/tests" -q
            WORKING_DIRECTORY "${ENTROPIC_PY_STAGE}")
        set_tests_properties(python_pytest PROPERTIES
            ENVIRONMENT "PYTHONPATH=${ENTROPIC_PY_STAGE}")
    endif()
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
