#define GL_GLEXT_PROTOTYPES
#include <glut.h>
#include <glext.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <iostream>

#include "cudanoise.cuh"

#define DIM 512

uchar4 *devPtr;
dim3 blocks(DIM / 16, DIM / 16);
dim3 threads(16, 16);
float zoom = 1.0f;

GLuint bufferObj;
cudaGraphicsResource *resource;

__global__ void kernel(uchar4 *ptr, float zoomFactor, int samples = 4)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float fx = x / (float)DIM + 24.234f;
	float fy = y / (float)DIM + 92.324f;

	float3 pos = make_float3(fx, fy, 0.0f);
	pos = scaleVector(pos, zoomFactor);

	float acc = 0.0f;

	for (int i = 0; i < samples; i++)
	{
		float dx = getRandomValue(327482 + i * 2347)  / (float)DIM * zoomFactor - 0.5f;
		float dy = getRandomValue(912472 + i * 118438)  / (float)DIM * zoomFactor - 0.5f;
		float dz = getRandomValue(112348 + i * 68214)  / (float)DIM * zoomFactor - 0.5f;

		float3 ditheredPos = make_float3(pos.x + dx, pos.y + dy, pos.z + dz);

		//	float val = checker(fx, fy, 0.0f, 64.0f);
		//	float val = discreteNoise(fx, fy, 0.0f, zoomFactor);
		//	float val = perlinNoise(ditheredPos);
		//	float val = repeater(ditheredPos, 1, 2.0f, 0.5f);
//			float val = turbulence(ditheredPos, 50.5f);
		float val = repeaterTurbulence(ditheredPos, 50.5f, 16);
//		float val = recursiveTurbulence(ditheredPos, 3, 2.0f, 0.5f, 1.0f);
//		float val = recursiveRepeaterTurbulence(ditheredPos, 4, 8, 2.0f, 0.5f, 1.0f);

		acc += val;
	}

	acc /= (float)samples;

	acc = mapToUnsigned(acc);
	acc = clamp(acc, 0.0f, 1.0f);

	unsigned char iVal = 255 * acc;

	ptr[offset].x = iVal;
	ptr[offset].y = iVal;
	ptr[offset].z = iVal;
	ptr[offset].w = 255;
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
	case 27:
		cudaGraphicsUnregisterResource(resource);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bufferObj);
		exit(0);
	case 43:
		std::cout << "Zoom requested." << std::endl;
		break;

	}
}

int main(int argc, char **argv)
{
	cudaDeviceProp prop;
	int dev;

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

	kernel << <blocks, threads >> > (devPtr, zoom);

	cudaGraphicsUnmapResources(1, &resource, NULL);

	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);
	glutMainLoop();
}