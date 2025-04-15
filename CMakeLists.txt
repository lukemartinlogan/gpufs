cmake_minimum_required(VERSION 3.20)
project(GPUFS)

# -----------------------------------------------------------------------------
# Define Options
# -----------------------------------------------------------------------------
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)
option(GPUFS_ENABLE_CMAKE_DOTENV "Load environment variables from .env.cmake" OFF)

# ------------------------------------------------------------------------------
# DOTENV
# ------------------------------------------------------------------------------
if(GPUFS_ENABLE_CMAKE_DOTENV)
    if(EXISTS ${CMAKE_SOURCE_DIR}/.env.cmake)
        include(.env.cmake)
    endif()
endif()

# -----------------------------------------------------------------------------
# Compiler Optimization
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("IN DEBUG MODE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")
    add_compile_definitions(HERMES_LOG_VERBOSITY=10)
else()
    message("IN RELEASE MODE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3")
    add_compile_definitions(HERMES_LOG_VERBOSITY=1)
endif()

add_compile_options(-march=native -fomit-frame-pointer)

# -----------------------------------------------------------------------------
# Targets built within this project are exported at Install time for use
# by other projects.
# -----------------------------------------------------------------------------
if(NOT CHIMAERA_EXPORTED_TARGETS)
    set(CHIMAERA_EXPORTED_TARGETS "GPUFS")
endif()

# ------------------------------------------------------------------------------
# Setup install and output Directories
# ------------------------------------------------------------------------------
if(NOT GPUFS_INSTALL_BIN_DIR)
    set(GPUFS_INSTALL_BIN_DIR ${CMAKE_INSTALL_PREFIX}/bin)
endif()

if(NOT GPUFS_INSTALL_LIB_DIR)
    set(GPUFS_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
endif()

if(NOT GPUFS_INSTALL_INCLUDE_DIR)
    set(GPUFS_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
endif()

if(NOT GPUFS_INSTALL_DATA_DIR)
    set(GPUFS_INSTALL_DATA_DIR ${CMAKE_INSTALL_PREFIX}/share)
endif()

# -----------------------------------------------------------------------------
# Find Packages
# -----------------------------------------------------------------------------

# This is for compatability with CLANG + Spack
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_INSTALL_RPATH "${GPUFS_INSTALL_LIB_DIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Include most packages needed by both client and host
find_package(HermesShm REQUIRED)

# ENABLE GPU SUPPORT
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode code=sm_75,arch=compute_75")
hshm_enable_cuda(11)

# ------------------------------------------------------------------------------
# Setup CMake Environment
# ------------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables.")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Libraries")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all static libraries.")

# -----------------------------------------------------------------------------
# Build Hermes Main Packages
# -----------------------------------------------------------------------------
# Main includes
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/libgpufs)
include_directories(${CMAKE_SOURCE_DIR}/gpufs_con_lib.user)

add_subdirectory(libgpufs)

# add_subdirectory(gpufs_con_lib.user)

# -----------------------------------------------------------------------------
# Install Hermes Headers
# -----------------------------------------------------------------------------
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX})

# jarvis_repo_add("${CMAKE_SOURCE_DIR}/test/jarvis_GPUFS" "${CMAKE_SOURCE_DIR}/test/jarvis_GPUFS/pipelines")