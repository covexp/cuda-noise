#include <iostream>
#include <chrono>
#include <fstream>
#include <string_view>

#include "../../include/cuda_noise.cuh"

__global__ void benchmarkPerlin(unsigned char* outputBuffer, int iterations)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    long idx = x + y * blockDim.x *  gridDim.x;

    float fx = static_cast<float>(x) / (blockDim.x * gridDim.x) * 16.0f;
    float fy = static_cast<float>(y) / (blockDim.y * gridDim.y) * 16.0f;

    float3 pos = make_float3(fx, fy, 0.0f);

    float sum = 0.0f;
    unsigned int seed = 0x71889283;
    for(int i = 0; i < iterations; i++)
    {
        seed = seed ^ ((i + 91482) * 1778932);
        sum += cudaNoise::repeaterPerlin(pos, 1.0f, seed, 32, 2.0f, 0.5f);
    }

    outputBuffer[idx] = static_cast<unsigned char>((sum / static_cast<float>(iterations)) * 63.0f + 127.0f);
}

__global__ void benchmarkSimplex(unsigned char* outputBuffer, int iterations)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    long idx = x + y * blockDim.x *  gridDim.x;

    float fx = static_cast<float>(x) / (blockDim.x * gridDim.x) * 16.0f;
    float fy = static_cast<float>(y) / (blockDim.y * gridDim.y) * 16.0f;

    float3 pos = make_float3(fx, fy, 0.0f);

    float sum = 0.0f;
    unsigned int seed = 0x71889283;
    for(int i = 0; i < iterations; i++)
    {
        seed = seed ^ ((i + 91482) * 1778932);
        sum += cudaNoise::repeaterSimplex(pos, 1.0f, seed, 32, 2.0f, 0.5f);
    }

    outputBuffer[idx] = static_cast<unsigned char>((sum / static_cast<float>(iterations)) * 127.0f + 127.0f);
}

void writeToDisk(unsigned char* buffer, const std::string& filename, size_t datasize)
{
    std::fstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<char*>(buffer), datasize);
}

int main()
{
    std::cout << "Benchmarking cuda-noise..." << std::endl;

    const size_t DIM = 4096;
    const int iterations = 32;

    dim3 blockSize {16, 16};
    dim3 gridSize { static_cast<int>(DIM) / blockSize.x, static_cast<int>(DIM) / blockSize.y };

    unsigned char* d_outputBuffer;
    unsigned char* h_outputBuffer;

    cudaMalloc((void**)&d_outputBuffer, DIM * DIM * sizeof(unsigned char));
    cudaMallocHost((void**)&h_outputBuffer, DIM * DIM * sizeof(unsigned char));

    {
        auto start = std::chrono::system_clock::now();
        benchmarkPerlin<<<gridSize, blockSize>>>(d_outputBuffer, iterations);
        cudaDeviceSynchronize();
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Perlin noise: " << elapsed.count() << " milliseconds" << std::endl;
    }

    cudaMemcpy(h_outputBuffer, d_outputBuffer, DIM * DIM * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    writeToDisk(h_outputBuffer, "perlin.data", DIM * DIM * sizeof(unsigned char));

    {
        auto start = std::chrono::system_clock::now();
        benchmarkSimplex<<<gridSize, blockSize>>>(d_outputBuffer, iterations);
        cudaDeviceSynchronize();
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Simplex noise: " << elapsed.count() << " milliseconds" << std::endl;
    }

    cudaMemcpy(h_outputBuffer, d_outputBuffer, DIM * DIM * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    writeToDisk(h_outputBuffer, "simplex.data", DIM * DIM * sizeof(unsigned char));

    cudaFree(d_outputBuffer);
    cudaFreeHost(h_outputBuffer);

    return 0;
}
