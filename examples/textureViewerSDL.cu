// textureViewer SDL

#include <cuda.h>
#include <iostream>

#include </usr/include/SDL2/SDL.h>

#include "../src/cudaNoise.cuh"

#define DIM 512
#define SIZE DIM * DIM

__global__ void kernel(uchar4 *buffer, float zoomFactor, int samples, int seed, int noise)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float fx = x / (float)DIM;
	float fy = y / (float)DIM;

	fx += 74.824f;
	fy += 38.234f;

	float3 pos = make_float3(fx, fy, 0.0f);
	pos = cudaNoise::scaleVector(pos, zoomFactor);

	float acc = 0.0f;

	float du = 1.0f / ((float)DIM / zoomFactor);

	for (int i = 0; i < samples; i++)
	{
		float dx = cudaNoise::randomFloat(327482 + i * 2347 + seed)  / (float)DIM * zoomFactor;
		float dy = cudaNoise::randomFloat(912472 + i * 118438 + seed)  / (float)DIM * zoomFactor;
		float dz = cudaNoise::randomFloat(112348 + i * 68214 + seed)  / (float)DIM * zoomFactor;

		float3 ditheredPos = make_float3(pos.x + dx, pos.y + dy, pos.z + dz);

		float val = 0.0f;

		
		switch (noise)
		{
		case(0):
			val = cudaNoise::perlinNoise(ditheredPos, 1.0f, seed);
			break;
		case(1):
			val = cudaNoise::simplexNoise(ditheredPos, 1.0f, seed);
			break;
		case(2):
			val = cudaNoise::worleyNoise(ditheredPos, 1.0f, seed, 300.1f, 4, 4, 1.0f);
			break;
		case(3):
			val = cudaNoise::repeaterPerlin(ditheredPos, 1.0f, seed, 128, 1.9f, 0.5f);
			break;
		case(4):
			val = cudaNoise::repeaterPerlinAbs(ditheredPos, 1.0f, seed, 128, 1.9f, 0.5f);
			break;
		case(5):
			val = cudaNoise::fractalSimplex(ditheredPos, 1.0f, seed, du, 512, 1.5f, 0.95f);
			break;
		case(6):
			val = cudaNoise::repeaterTurbulence(ditheredPos, 0.2f, 1.0f, seed, 0.8f, 32, cudaNoise::BASIS_PERLIN, cudaNoise::BASIS_PERLIN);
			break;
		case(7):
			val = cudaNoise::cubicValue(ditheredPos, 1.0f, seed);
			break;
		case(8):
			val = cudaNoise::spots(ditheredPos, 1.0f, seed, 0.1f, 0, 8, 1.0f, cudaNoise::SHAPE_STEP);
			break;
		}

		acc += val;
	}

	acc /= (float)samples;

	acc = cudaNoise::mapToUnsigned(acc);
	acc = cudaNoise::clamp(acc, 0.0f, 1.0f);

	unsigned char iVal = 255 * acc;

	buffer[offset].x = iVal;
	buffer[offset].y = iVal;
	buffer[offset].z = iVal;
	buffer[offset].w = 255;
}

int main(int argc, char **argv)
{
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);

	cudaDeviceProp prop;
	int dev;

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice(&dev, &prop);

	uchar4 *h_buffer = new uchar4[SIZE];

	uchar4 *d_buffer;
	cudaMalloc((void**) &d_buffer, SIZE * sizeof(uchar4));

	kernel << <blocks, threads >> > (d_buffer, 1.0, 1, 42, 1);

	delete[] h_buffer;

	cudaFree(d_buffer);
}
