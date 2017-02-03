// cudanoise
// Library of common 3D noise functions for CUDA kernels

#define N 512
#define WORLDSIZE N * N

#include <cuda_runtime.h>
#include "cudanoise.cuh"

__device__ unsigned int hash(unsigned int a)
{
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

__device__ float getRandomValue(unsigned int seed)
{
	unsigned int noiseVal = hash(seed);
	return ((float)noiseVal / (float)0xffffffff);
}

__device__ float clamp(float val, float min, float max)
{
	if (val < 0.0f)
		return 0.0f;
	else if (val > 1.0f)
		return 1.0f;

	return val;
}

__device__ float mapToSigned(float input)
{
	return input * 2.0f - 1.0f;
}

__device__ float mapToUnsigned(float input)
{
	return input * 0.5f + 0.5f;
}

__device__ float checker(float x, float y, float z, float scale)
{
	int ix = (int)(x * scale);
	int iy = (int)(y * scale);
	int iz = (int)(z * scale);

	if ((ix + iy + iz) % 2 == 0)
		return 1.0f;

	return 0.0f;
}

__device__ float whiteNoise(int x, int y, int z)
{
	return getRandomValue(x * 231 + y * 91023 + z * 48172);
}

__device__ float3 vectorNoise(int x, int y, int z)
{
	return make_float3(getRandomValue(x * 8231 + y * 34612 + z * 11836 + 19283) * 2.0f - 1.0f,
		   			   getRandomValue(x * 1171 + y * 9234 + z * 992903 + 1466) * 2.0f - 1.0f,
					   0.0f);
}

__device__ float3 scaleVector(float3 v, float factor)
{
	return make_float3(v.x * factor, v.y * factor, v.z * factor);
}

__device__ float3 addVectors(float3 v, float3 w)
{
	return make_float3(v.x + w.x, v.y + w.y, v.z + w.z);
}

__device__ float dotProduct(float3 u, float3 v)
{
	return (u.x * v.x + u.y * v.y + u.z * v.z);
}

__device__ float lerp(float a, float b, float ratio)
{
	return a * (1.0f - ratio) + b * ratio;
}

__device__ float discreteNoise(float x, float y, float z, float scale)
{
	int ix = (int)(x * scale);
	int iy = (int)(y * scale);
	int iz = (int)(z * scale);

	return whiteNoise(ix, iy, iz);
}

__device__ float grad(int hash, float x, float y, float z)
{
	switch (hash & 0xF)
	{
		case 0x0: return  x + y;
		case 0x1: return -x + y;
		case 0x2: return  x - y;
		case 0x3: return -x - y;
		case 0x4: return  x + z;
		case 0x5: return -x + z;
		case 0x6: return  x - z;
		case 0x7: return -x - z;
		case 0x8: return  y + z;
		case 0x9: return -y + z;
		case 0xA: return  y - z;
		case 0xB: return -y - z;
		case 0xC: return  y + x;
		case 0xD: return -y + z;
		case 0xE: return  y - x;
		case 0xF: return -y - z;
	default: return 0; // never happens
	}
}

__device__ int getHash(int x, int y, int z)
{
	return hash((unsigned int)(x * 1723 + y * 93241 + z * 149812 + 3824));
}

__device__ float fade(float t) 
{
	// Fade function as defined by Ken Perlin.  This eases coordinate values
	// so that they will ease towards integral values.  This ends up smoothing
	// the final output.
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);         // 6t^5 - 15t^4 + 10t^3
}

__device__ float perlinNoise(float3 pos)
{
	// zero corner integer position
	int ix = (int)floorf(pos.x);
	int iy = (int)floorf(pos.y);
	int iz = (int)floorf(pos.z);

	// current position within unit cube
	pos.x -= floorf(pos.x);
	pos.y -= floorf(pos.y);
	pos.z -= floorf(pos.z);

	// adjust for fade
	float u = fade(pos.x);
	float v = fade(pos.y);
	float w = fade(pos.z);

	// influence values
	float i000 = grad(getHash(ix, iy, iz), pos.x, pos.y, pos.z);
	float i100 = grad(getHash(ix + 1, iy, iz), pos.x - 1.0f, pos.y, pos.z);
	float i010 = grad(getHash(ix, iy + 1, iz), pos.x, pos.y - 1.0f, pos.z);
	float i110 = grad(getHash(ix + 1, iy + 1, iz), pos.x - 1.0f, pos.y - 1.0f, pos.z);
	float i001 = grad(getHash(ix, iy, iz + 1), pos.x, pos.y, pos.z - 1.0f);
	float i101 = grad(getHash(ix + 1, iy, iz + 1), pos.x - 1.0f, pos.y, pos.z - 1.0f);
	float i011 = grad(getHash(ix, iy + 1, iz + 1), pos.x, pos.y - 1.0f, pos.z - 1.0f);
	float i111 = grad(getHash(ix + 1, iy + 1, iz + 1), pos.x - 1.0f, pos.y - 1.0f, pos.z - 1.0f);

	// interpolation
	float x00 = lerp(i000, i100, u);
	float x10 = lerp(i010, i110, u);
	float x01 = lerp(i001, i101, u);
	float x11 = lerp(i011, i111, u);

	float y0 = lerp(x00, x10, v);
	float y1 = lerp(x01, x11, v);

	float avg = lerp(y0, y1, w);

	return avg;
}

__device__ float repeater(float3 pos, int n, float harmonic = 2.0f, float decay = 0.5f)
{
	float scale = 1.0f;
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		acc += perlinNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale)) * amp;
		scale *= harmonic;
		amp *= decay;
	}

	return acc;
}

__device__ float turbulence(float3 pos, float strength)
{
	pos.x += perlinNoise(pos) * strength;

	return perlinNoise(pos);
}

__device__ float repeaterTurbulence(float3 pos, float strength, int n)
{
	pos.x += (repeater(make_float3(pos.x, pos.y, pos.z), n)) * strength;

	return repeater(pos, n);
}

__device__ float recursiveTurbulence(float3 pos, int n, float harmonic = 2.0f, float decay = 0.5f, float strength = 1.0f)
{
	float3 displace = make_float3(pos.x, pos.y, pos.z);
	float scale = 1.0f;
	float amp = 1.0f;
	float acc = 0.0f;

	for (int i = 0; i < n; i++)
	{
		acc += perlinNoise(scaleVector(displace, scale)) * amp;

		displace.x += perlinNoise(make_float3(pos.x, pos.y, pos.z)) * amp * strength;
		displace.y += acc * strength;
		displace.z += perlinNoise(make_float3(acc, acc, acc)) * amp * strength;

		scale *= harmonic;
		amp *= decay;
	}

	return acc / 1.0f;
}

__device__ float recursiveRepeaterTurbulence(float3 pos, int n, int m, float harmonic = 2.0f, float decay = 0.5f, float strength = 1.0f)
{
	float3 displace = make_float3(pos.x, pos.y, pos.z);
	float scale = 1.0f;
	float amp = 1.0f;
	float acc = 0.0f;

	for (int i = 0; i < n; i++)
	{
		acc += repeater(scaleVector(displace, scale), m) * amp;

		displace.x += repeater(make_float3(pos.x, pos.y, pos.z), m) * amp * strength;
		displace.y += acc * strength;
		displace.z += repeater(make_float3(acc, acc, acc), m) * amp * strength;

		scale *= harmonic;
		amp *= decay;
	}

	return acc / 1.0f;
}