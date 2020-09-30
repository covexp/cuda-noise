cmake_minimum_required(VERSION 3.17)
project(benchmark CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61 -use_fast_math --ptxas-options=-v,-O3")
message(NVCC Flags: ${CMAKE_CUDA_FLAGS})

add_executable(benchmark main.cu)

set_target_properties(
        benchmark
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)