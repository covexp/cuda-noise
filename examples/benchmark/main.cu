#include <iostream>
#include <chrono>

#include "../../include/cuda_noise.cuh"

__global__ void benchmarkPerlin(float* outputBuffer, int iterations)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    long idx = x + y * blockDim.x *  gridDim.x;

    float fx = static_cast<float>(x) / (blockDim.x * gridDim.x);
    float fy = static_cast<float>(y) / (blockDim.y * gridDim.y);

    float3 pos = make_float3(fx, fy, 0.0f);

    unsigned int seed = 0x71889283;
    for(int i = 0; i < iterations; i++)
    {
        seed = seed ^ ((i + 91482) * 1778932);
        outputBuffer[idx] = cudaNoise::repeaterPerlin(pos, 1.0f, seed, 32, 2.0f, 0.5f);
    }
}

__global__ void benchmarkSimplex(float* outputBuffer, int iterations)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    long idx = x + y * blockDim.x *  gridDim.x;

    float fx = static_cast<float>(x) / (blockDim.x * gridDim.x);
    float fy = static_cast<float>(y) / (blockDim.y * gridDim.y);

    float3 pos = make_float3(fx, fy, 0.0f);

    unsigned int seed = 0x71889283;
    for(int i = 0; i < iterations; i++)
    {
        seed = seed ^ ((i + 91482) * 1778932);
        outputBuffer[idx] = cudaNoise::repeaterSimplex(pos, 1.0f, seed, 32, 2.0f, 0.5f);
    }
}

int main() {
    std::cout << "Benchmarking cuda-noise..." << std::endl;

    const size_t DIM = 4096;

    dim3 blockSize {16, 16};
    dim3 gridSize { static_cast<int>(DIM) / blockSize.x, static_cast<int>(DIM) / blockSize.y };

    float* d_outputBuffer;
    float* h_outputBuffer;

    cudaMalloc((void**)&d_outputBuffer, DIM * DIM * sizeof(float));
    cudaMallocHost((void**)&h_outputBuffer, DIM * DIM * sizeof(float));

    {
        auto start = std::chrono::system_clock::now();
        benchmarkPerlin<<<gridSize, blockSize>>>(d_outputBuffer, 32);
        cudaDeviceSynchronize();
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Perlin noise: " << elapsed.count() << " milliseconds" << std::endl;
    }

    {
        auto start = std::chrono::system_clock::now();
        benchmarkSimplex<<<gridSize, blockSize>>>(d_outputBuffer, 32);
        cudaDeviceSynchronize();
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Simplex noise: " << elapsed.count() << " milliseconds" << std::endl;
    }

    cudaMemcpy(h_outputBuffer, d_outputBuffer, DIM * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_outputBuffer);
    cudaFreeHost(h_outputBuffer);

    return 0;
}
