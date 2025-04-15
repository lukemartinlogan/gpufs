cmake_minimum_required(VERSION 3.18)

# Find CUDA
find_package(CUDA REQUIRED)

# Collect all source files
file(GLOB SOURCES
    "*.cu"
    "*.cpp"
)

# Create the library target
add_cuda_library(gpufs STATIC FALSE ${SOURCES})

# Include directories if needed
target_include_directories(gpufs PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)