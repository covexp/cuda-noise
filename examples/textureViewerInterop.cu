// textureViewer
// Simple GL texture viewer to preview 2D slices of textures produced by cudaNoise

#define GL_GLEXT_PROTOTYPES
#include <glut.h>
#include <glext.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <time.h>

#include "cudanoise.cuh"

#define DIM 512

uchar4 *devPtr;
dim3 blocks(DIM / 16, DIM / 16);
dim3 threads(16, 16);

float zoom = 16.0f;
int genSeed = 0;
int selectedNoise = 0;

GLuint bufferObj;
cudaGraphicsResource *resource;

__global__ void kernel(uchar4 *ptr, float zoomFactor, int samples, int seed, int noise)
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
		

//		val = cudaNoise::checker(ditheredPos, 1.0f, seed);
//		val = cudaNoise::discreteNoise(ditheredPos, 1.0f, 3478);
//		val = cudaNoise::linearValue(ditheredPos, 1.0f, seed);
//		val = cudaNoise::perlinNoise(ditheredPos, 1.0f, seed);
//		val = cudaNoise::simplexNoise(ditheredPos, 0.01f, seed);
//		val = cudaNoise::worleyNoise(ditheredPos, 1.0f, seed, 300.1f, 4, 4, 1.0f);
//		val = cudaNoise::repeater(ditheredPos, 1.0f, seed, 4, 1.5f, 0.75f, cudaNoise::BASIS_PERLIN);
//		val = cudaNoise::repeaterPerlin(ditheredPos, 1.0f, seed, 128, 1.9f, 0.5f);
//		val = cudaNoise::repeaterPerlinAbs(ditheredPos, 1.0f, seed, 128, 1.9f, 0.5f);
//		val = cudaNoise::repeaterSimplex(ditheredPos, 1.0f, seed, 128, 1.5f, 0.8f);
//		val = cudaNoise::repeaterSimplexAbs(ditheredPos, 1.0f, seed, 16, 1.5f, 0.8f);
//		val = cudaNoise::fractalSimplex(ditheredPos, 1.0f, seed, du, 512, 1.5f, 0.95f);
//		val = cudaNoise::turbulence(ditheredPos, 4.0f, 1.0f, seed, 0.02f, cudaNoise::BASIS_PERLIN, cudaNoise::BASIS_CHECKER);
//		val = cudaNoise::repeaterTurbulence(ditheredPos, 0.2f, 1.0f, seed, 0.8f, 32, CUDANOISE_PERLIN, CUDANOISE_PERLIN);
//		val = cudaNoise::cubicValue(ditheredPos, 1.0f, seed);
//		val = cudaNoise::fadedValue(ditheredPos, 1.0f);
//		val = cudaNoise::spots(ditheredPos, 1.0f, seed, 0.1f, 0, 8, 1.0f, cudaNoise::SHAPE_STEP);

		acc += val;
	}

	acc /= (float)samples;

	acc = cudaNoise::mapToUnsigned(acc);
	acc = cudaNoise::clamp(acc, 0.0f, 1.0f);

	unsigned char iVal = 255 * acc;

	ptr[offset].x = iVal;
	ptr[offset].y = iVal;
	ptr[offset].z = iVal;
	ptr[offset].w = 255;
}

void setSeed(int newSeed)
{
	genSeed = newSeed;
}

void redrawTexture()
{
	time_t startTime = clock();

	cudaGraphicsMapResources(1, &resource, NULL);
	kernel << < blocks, threads >> > (devPtr, zoom, 1, genSeed, selectedNoise);	
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &resource, NULL);

	time_t endTime = clock();

	double time_spent = (double)(endTime - startTime) / CLOCKS_PER_SEC;

	printf("Time spent: %f\n", time_spent);

	glutPostRedisplay();
}

static void idle_func(void)
{
}

static void draw_func(void)
{
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

static void key_func(unsigned char key, int x, int y)
{
	switch (key)
	{
	// ESC to exit
	case 27:
		cudaGraphicsUnregisterResource(resource);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bufferObj);
		exit(0);
		break;
	// Plus to zoom in
	case 43:
		zoom *= 0.5f;
		redrawTexture();
		break;
	// Minus to zoom out
	case 45:
		zoom *= 2.0f;
		redrawTexture();
		break;
	// Dot to get the next noise function
	case 46:
		std::cout << "KEA" << std::endl;
		selectedNoise = (selectedNoise + 1) % 9;
		redrawTexture();
		break;
	// Spacebar to get new seed
	case 32:
		clock_t t = clock();
		unsigned int newSeed = (unsigned int)((double)t * 1000.0f);
		setSeed(newSeed);
		redrawTexture();
		break;
	}
}

int main(int argc, char **argv)
{
	cudaDeviceProp prop;
	int dev;

	setSeed(time(NULL));

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice(&dev, &prop);

	cudaGLSetGLDevice(dev);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("cudaNoise - Texture Viewer");

	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);

	size_t size;
	cudaGraphicsMapResources(1, &resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

	kernel << <blocks, threads >> > (devPtr, zoom, 1, genSeed, selectedNoise);

	cudaGraphicsUnmapResources(1, &resource, NULL);

	glutIdleFunc(idle_func);
	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);
	glutMainLoop();

	printf("\n\n");
}