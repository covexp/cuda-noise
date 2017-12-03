// textureViewer SDL

#include <cuda.h>
#include <iostream>
#include <ctime>

#include </usr/include/SDL2/SDL.h>

#include "../include/cuda_noise.cuh"

const int DIM = 512;
const int SIZE = DIM * DIM;

__global__ void kernel(Uint32 *buffer, float zoomFactor, int samples, int seed, int noise)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float fx = x / (float)DIM;
	float fy = y / (float)DIM;

	fx += 724.824f;
	fy += 338.234f;

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
			val = cudaNoise::repeaterTurbulence(ditheredPos, 0.2f, 1.0f, seed, 0.8f, 12, cudaNoise::BASIS_PERLIN, cudaNoise::BASIS_PERLIN);
			break;
		case(7):
			val = cudaNoise::cubicValue(ditheredPos, 1.0f, seed);
			break;
		case(8):
			val = cudaNoise::spots(ditheredPos, 1.0f, seed, 0.1f, 0, 8, 1.0f, cudaNoise::SHAPE_STEP);
			break;
		case(9):
			val = cudaNoise::turbulence(ditheredPos, 10.2f, 12.120f, seed, 0.27f, cudaNoise::BASIS_SIMPLEX, cudaNoise::BASIS_SIMPLEX);
			break;
		}

		acc += val;
	}

	acc /= (float)samples;

	acc = cudaNoise::mapToUnsigned(acc);
	acc = cudaNoise::clamp(acc, 0.0f, 1.0f);

	unsigned char iVal = 255 * acc;

	Uint32 colorVal = iVal;
	colorVal += iVal << 8;
	colorVal += iVal << 16;

	buffer[offset] = colorVal;
}

void paintSDL(Uint32 *image)
{
	SDL_Window *window= NULL;
	SDL_Surface *screenSurface = NULL;

	if(SDL_Init(SDL_INIT_VIDEO) < 0)
		return;

	window = SDL_CreateWindow("cudaNoise Texture Viewer",
	                          SDL_WINDOWPOS_CENTERED,
	                          SDL_WINDOWPOS_CENTERED,
	                          DIM,
	                          DIM,
	                          SDL_WINDOW_SHOWN);

	if(window == NULL)
		return;

	screenSurface = SDL_GetWindowSurface(window);
	std::cout << "Bytes per pixel: " << (int) screenSurface->format->BytesPerPixel << std::endl;

	SDL_FillRect(screenSurface, NULL, SDL_MapRGB(screenSurface->format, 0xFF, 0xFF, 0xFF));

	Uint32 *pixels = (Uint32 *) screenSurface->pixels;

	for(int j = 0; j < DIM; j++)
	{
		for(int i = 0; i < DIM; i++)
		{
			int idx = i + j * DIM;
			pixels[idx] = image[idx];
		}
	}

	SDL_UpdateWindowSurface(window);
	SDL_Delay(4000);
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

	Uint32 *h_buffer = new Uint32[SIZE];

	Uint32 *d_buffer;
	cudaMalloc((void**) &d_buffer, SIZE * sizeof(Uint32));

	std::cout << "Running kernel..." << std::endl;

	clock_t timeBegin = clock();

	kernel << <blocks, threads >> > (d_buffer, 1.0, 16, 42, 9);
	cudaDeviceSynchronize();

	clock_t timeEnd = clock();

	unsigned int ticks = (unsigned int)(timeEnd - timeBegin);

	std::cout << "Run took: " << ticks << " ticks." << std::endl;

	cudaMemcpy(h_buffer, d_buffer, SIZE * sizeof(Uint32), cudaMemcpyDeviceToHost);

	paintSDL(h_buffer);

	delete[] h_buffer;

	cudaFree(d_buffer);
}
