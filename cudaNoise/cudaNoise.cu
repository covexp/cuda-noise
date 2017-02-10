// cudanoise
// Library of common 3D noise functions for CUDA kernels

#define N 512
#define WORLDSIZE N * N

#include <cuda_runtime.h>
#include "cudanoise.cuh"

__device__ unsigned int hash(unsigned int seed)
{
	seed = (seed + 0x7ed55d16) + (seed << 12);
	seed = (seed ^ 0xc761c23c) ^ (seed >> 19);
	seed = (seed + 0x165667b1) + (seed << 5);
	seed = (seed + 0xd3a2646c) ^ (seed << 9);
	seed = (seed + 0xfd7046c5) + (seed << 3);
	seed = (seed ^ 0xb55a4f09) ^ (seed >> 16);

	return seed;
}

__device__ int randomIntRange(int min, int max, int seed)
{
	int base = hash(seed);
	base = base % (1 + max - min) + min;

	return base;
}

__device__ float randomFloat(unsigned int seed)
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

// Random float for a grid coordinate [-1, 1]
__device__ float randomGrid(int x, int y, int z, int seed = 0)
{
	return mapToSigned(randomFloat((unsigned int)(x * 1723 + y * 93241 + z * 149812 + 3824 + seed)));
}

// Random unsigned int for a grid coordinate [0, MAXUINT]
__device__ unsigned int randomIntGrid(int x, int y, int z, int seed = 0)
{
	return hash((unsigned int)(x * 1723 + y * 93241 + z * 149812 + 3824 + seed));
}

__device__ float3 vectorNoise(int x, int y, int z)
{
	return make_float3(randomFloat(x * 8231 + y * 34612 + z * 11836 + 19283) * 2.0f - 1.0f,
		   			   randomFloat(x * 1171 + y * 9234 + z * 992903 + 1466) * 2.0f - 1.0f,
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

// Helper functions for noise noise

__device__ float lerp(float a, float b, float ratio)
{
	return a * (1.0f - ratio) + b * ratio;
}

__device__ float cubic(float p0, float p1, float p2, float p3, float x)
{
	return p1 + 0.5 * x * (p2 - p0 + x * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + x * (3.0 * (p1 - p2) + p3 - p0)));
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

__device__ float fade(float t)
{
	// Fade function as defined by Ken Perlin.  This eases coordinate values
	// so that they will ease towards integral values.  This ends up smoothing
	// the final output.
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);         // 6t^5 - 15t^4 + 10t^3
}

// Noise functions

__device__ float checker(float3 pos, float scale, int seed)
{
	int ix = (int)(pos.x * scale);
	int iy = (int)(pos.y * scale);
	int iz = (int)(pos.z * scale);

	if ((ix + iy + iz) % 2 == 0)
		return 1.0f;

	return 0.0f;
}

__device__ float spots(float3 pos, float scale, int seed, float size, int minNum, int maxNum, float jitter, profileShape shape)
{
	if (size < EPSILON)
		return 0.0f;

	int ix = (int)(pos.x * scale);
	int iy = (int)(pos.y * scale);
	int iz = (int)(pos.z * scale);

	float u = pos.x - (float)ix;
	float v = pos.y - (float)iy;
	float w = pos.z - (float)iz;

	float val = -1.0f;

	// We need to traverse the entire 3x3x3 neighborhood in case there are spots in neighbors near the edges of the cell
	for (int x = -1; x < 2; x++)
	{
		for (int y = -1; y < 2; y++)
		{
			for (int z = -1; z < 2; z++)
			{
				int numSpots = randomIntRange(minNum, maxNum, seed + (ix + x) * 823746 + (iy + y) * 12306 + (iz + z) * 823452 + 3234874);

				for (int i = 0; i < numSpots; i++)
				{
					float distU = u - x - (randomFloat(seed + (ix + x) * 23784 + (iy + y) * 9183 + (iz + z) * 23874 + 334 * i + 27432) * jitter - jitter / 2.0f);
					float distV = v - y - (randomFloat(seed + (ix + x) * 12743 + (iy + y) * 45191 + (iz + z) * 144421 + 2934 * i + 76671) * jitter - jitter / 2.0f);
					float distW = w - z - (randomFloat(seed + (ix + x) * 82734 + (iy + y) * 900213 + (iz + z) * 443241 + 18237 * i + 199823) * jitter - jitter / 2.0f);

					float distanceSq = distU * distU + distV * distV + distW * distW;

					switch (shape)
					{
					case(CUDANOISE_STEP):
						if (distanceSq < size)
							val = fmaxf(val, 1.0f);
						else
							val = fmaxf(val, -1.0f);
						break;
					case(CUDANOISE_LINEAR):
						float distanceAbs = fabsf(distU) + fabsf(distV) + fabsf(distW);
						val = fmaxf(val, 1.0f - clamp(distanceAbs, 0.0f, size) / size);
						break;
					case(CUDANOISE_QUADRATIC):
						val = fmaxf(val, 1.0f - clamp(distanceSq, 0.0f, size) / size);
						break;
					}
				}
			}
		}
	}

	return val;
}

__device__ float tricubic(int x, int y, int z, float u, float v, float w)
{
	// interpolate along x first
	float x00 = cubic(randomGrid(x - 1, y - 1, z - 1), randomGrid(x, y - 1, z - 1), randomGrid(x + 1, y - 1, z - 1), randomGrid(x + 2, y - 1, z - 1), u);
	float x01 = cubic(randomGrid(x - 1, y - 1, z), randomGrid(x, y - 1, z), randomGrid(x + 1, y - 1, z), randomGrid(x + 2, y - 1, z), u);
	float x02 = cubic(randomGrid(x - 1, y - 1, z + 1), randomGrid(x, y - 1, z + 1), randomGrid(x + 1, y - 1, z + 1), randomGrid(x + 2, y - 1, z + 1), u);
	float x03 = cubic(randomGrid(x - 1, y - 1, z + 2), randomGrid(x, y - 1, z + 2), randomGrid(x + 1, y - 1, z + 2), randomGrid(x + 2, y - 1, z + 2), u);

	float x10 = cubic(randomGrid(x - 1, y, z - 1), randomGrid(x, y, z - 1), randomGrid(x + 1, y, z - 1), randomGrid(x + 2, y, z - 1), u);
	float x11 = cubic(randomGrid(x - 1, y, z), randomGrid(x, y, z), randomGrid(x + 1, y, z), randomGrid(x + 2, y, z), u);
	float x12 = cubic(randomGrid(x - 1, y, z + 1), randomGrid(x, y, z + 1), randomGrid(x + 1, y, z + 1), randomGrid(x + 2, y, z + 1), u);
	float x13 = cubic(randomGrid(x - 1, y, z + 2), randomGrid(x, y, z + 2), randomGrid(x + 1, y, z + 2), randomGrid(x + 2, y, z + 2), u);

	float x20 = cubic(randomGrid(x - 1, y + 1, z - 1), randomGrid(x, y + 1, z - 1), randomGrid(x + 1, y + 1, z - 1), randomGrid(x + 2, y + 1, z - 1), u);
	float x21 = cubic(randomGrid(x - 1, y + 1, z), randomGrid(x, y + 1, z), randomGrid(x + 1, y + 1, z), randomGrid(x + 2, y + 1, z), u);
	float x22 = cubic(randomGrid(x - 1, y + 1, z + 1), randomGrid(x, y + 1, z + 1), randomGrid(x + 1, y + 1, z + 1), randomGrid(x + 2, y + 1, z + 1), u);
	float x23 = cubic(randomGrid(x - 1, y + 1, z + 2), randomGrid(x, y + 1, z + 2), randomGrid(x + 1, y + 1, z + 2), randomGrid(x + 2, y + 1, z + 2), u);

	float x30 = cubic(randomGrid(x - 1, y + 2, z - 1), randomGrid(x, y + 2, z - 1), randomGrid(x + 1, y + 2, z - 1), randomGrid(x + 2, y + 2, z - 1), u);
	float x31 = cubic(randomGrid(x - 1, y + 2, z), randomGrid(x, y + 2, z), randomGrid(x + 1, y + 2, z), randomGrid(x + 2, y + 2, z), u);
	float x32 = cubic(randomGrid(x - 1, y + 2, z + 1), randomGrid(x, y + 2, z + 1), randomGrid(x + 1, y + 2, z + 1), randomGrid(x + 2, y + 2, z + 1), u);
	float x33 = cubic(randomGrid(x - 1, y + 2, z + 2), randomGrid(x, y + 2, z + 2), randomGrid(x + 1, y + 2, z + 2), randomGrid(x + 2, y + 2, z + 2), u);

	// interpolate along y
	float y0 = cubic(x00, x10, x20, x30, v);
	float y1 = cubic(x01, x11, x21, x31, v);
	float y2 = cubic(x02, x12, x22, x32, v);
	float y3 = cubic(x03, x13, x23, x33, v);

	// interpolate along z
	return cubic(y0, y1, y2, y3, w);
}

__device__ float discreteNoise(float3 pos, float scale, int seed)
{
	int ix = (int)(pos.x * scale);
	int iy = (int)(pos.y * scale);
	int iz = (int)(pos.z * scale);

	return randomGrid(ix, iy, iz, seed);
}

__device__ float linearValue(float3 pos, float scale, int seed)
{
	int ix = (int)pos.x;
	int iy = (int)pos.y;
	int iz = (int)pos.z;

	float u = pos.x - ix;
	float v = pos.y - iy;
	float w = pos.z - iz;

	// Corner values
	float a000 = randomGrid(ix, iy, iz, seed);
	float a100 = randomGrid(ix + 1, iy, iz, seed);
	float a010 = randomGrid(ix, iy + 1, iz, seed);
	float a110 = randomGrid(ix + 1, iy + 1, iz, seed);
	float a001 = randomGrid(ix, iy, iz + 1, seed);
	float a101 = randomGrid(ix + 1, iy, iz + 1, seed);
	float a011 = randomGrid(ix, iy + 1, iz + 1, seed);
	float a111 = randomGrid(ix + 1, iy + 1, iz + 1, seed);

	// Linear interpolation
	float x00 = lerp(a000, a100, u);
	float x10 = lerp(a010, a110, u);
	float x01 = lerp(a001, a101, u);
	float x11 = lerp(a011, a111, u);

	float y0 = lerp(x00, x10, v);
	float y1 = lerp(x01, x11, v);

	return lerp(y0, y1, w);
}

__device__ float fadedValue(float3 pos, float scale, int seed)
{
	int ix = (int)(pos.x * scale);
	int iy = (int)(pos.y * scale);
	int iz = (int)(pos.z * scale);

	float u = fade(pos.x - ix);
	float v = fade(pos.y - iy);
	float w = fade(pos.z - iz);

	// Corner values
	float a000 = randomGrid(ix, iy, iz);
	float a100 = randomGrid(ix + 1, iy, iz);
	float a010 = randomGrid(ix, iy + 1, iz);
	float a110 = randomGrid(ix + 1, iy + 1, iz);
	float a001 = randomGrid(ix, iy, iz + 1);
	float a101 = randomGrid(ix + 1, iy, iz + 1);
	float a011 = randomGrid(ix, iy + 1, iz + 1);
	float a111 = randomGrid(ix + 1, iy + 1, iz + 1);

	// Linear interpolation
	float x00 = lerp(a000, a100, u);
	float x10 = lerp(a010, a110, u);
	float x01 = lerp(a001, a101, u);
	float x11 = lerp(a011, a111, u);

	float y0 = lerp(x00, x10, v);
	float y1 = lerp(x01, x11, v);

	return lerp(y0, y1, w) / 2.0f * 1.0f;
}

__device__ float cubicValue(float3 pos, float scale, int seed)
{
	pos.x = pos.x * scale;
	pos.y = pos.y * scale;
	pos.z = pos.z * scale;

	int ix = (int)pos.x;
	int iy = (int)pos.y;
	int iz = (int)pos.z;

	float u = pos.x - ix;
	float v = pos.y - iy;
	float w = pos.z - iz;

	return tricubic(ix, iy, iz, u, v, w);
}

__device__ float perlinNoise(float3 pos, float scale, int seed)
{
	pos.x = pos.x * scale;
	pos.y = pos.y * scale;
	pos.z = pos.z * scale;

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
	float i000 = grad(randomIntGrid(ix, iy, iz, seed), pos.x, pos.y, pos.z);
	float i100 = grad(randomIntGrid(ix + 1, iy, iz, seed), pos.x - 1.0f, pos.y, pos.z);
	float i010 = grad(randomIntGrid(ix, iy + 1, iz, seed), pos.x, pos.y - 1.0f, pos.z);
	float i110 = grad(randomIntGrid(ix + 1, iy + 1, iz, seed), pos.x - 1.0f, pos.y - 1.0f, pos.z);
	float i001 = grad(randomIntGrid(ix, iy, iz + 1, seed), pos.x, pos.y, pos.z - 1.0f);
	float i101 = grad(randomIntGrid(ix + 1, iy, iz + 1, seed), pos.x - 1.0f, pos.y, pos.z - 1.0f);
	float i011 = grad(randomIntGrid(ix, iy + 1, iz + 1, seed), pos.x, pos.y - 1.0f, pos.z - 1.0f);
	float i111 = grad(randomIntGrid(ix + 1, iy + 1, iz + 1, seed), pos.x - 1.0f, pos.y - 1.0f, pos.z - 1.0f);

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

__device__ float repeater(float3 pos, float scale, int seed, int n, float lacunarity, float decay, basisFunction basis)
{
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		switch (basis)
		{
		case(CUDANOISE_CHECKER):
			acc += checker(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
			break;
		case(CUDANOISE_LINEARVALUE):
			acc += linearValue(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
			break;
		case(CUDANOISE_FADEDVALUE):
			acc += fadedValue(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
			break;
		case(CUDANOISE_CUBICVALUE):
			acc += cubicValue(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
			break;
		case(CUDANOISE_PERLIN):
			acc += perlinNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
			break;
		}

		scale *= lacunarity;
		amp *= decay;
	}

	return acc;
}

__device__ float turbulence(float3 pos, float scaleIn, float scaleOut, int seed, float strength, basisFunction inFunc, basisFunction outFunc)
{
	switch (inFunc)
	{
	case(CUDANOISE_CHECKER):
		pos.x += checker(pos, scaleIn, seed) * strength;
		pos.y += checker(pos, scaleIn, seed) * strength;
		pos.z += checker(pos, scaleIn, seed) * strength;
		break;
	case(CUDANOISE_LINEARVALUE):
		pos.x += linearValue(pos, scaleIn, seed) * strength;
		pos.y += linearValue(pos, scaleIn, seed) * strength;
		pos.z += linearValue(pos, scaleIn, seed) * strength;
		break;
	case(CUDANOISE_FADEDVALUE):
		pos.x += fadedValue(pos, scaleIn, seed) * strength;
		pos.y += fadedValue(pos, scaleIn, seed) * strength;
		pos.z += fadedValue(pos, scaleIn, seed) * strength;
		break;
	case(CUDANOISE_CUBICVALUE):
		pos.x += cubicValue(pos, scaleIn, seed) * strength;
		pos.y += cubicValue(pos, scaleIn, seed) * strength;
		pos.z += cubicValue(pos, scaleIn, seed) * strength;
		break;
	case(CUDANOISE_PERLIN):
		pos.x += perlinNoise(pos, scaleIn, seed) * strength;
		pos.y += perlinNoise(pos, scaleIn, seed) * strength;
		pos.z += perlinNoise(pos, scaleIn, seed) * strength;
		break;
	}

	switch (outFunc)
	{
	case(CUDANOISE_CHECKER):
		return checker(pos, scaleOut, seed);
		break;
	case(CUDANOISE_LINEARVALUE):
		return linearValue(pos, scaleOut, seed);
		break;
	case(CUDANOISE_FADEDVALUE):
		return fadedValue(pos, scaleOut, seed);
		break;
	case(CUDANOISE_CUBICVALUE):
		return cubicValue(pos, scaleOut, seed);
		break;
	case(CUDANOISE_PERLIN):
		return perlinNoise(pos, scaleOut, seed);
		break;
	}

	return 0.0f;
}

__device__ float repeaterTurbulence(float3 pos, float scaleIn, float scaleOut, int seed, float strength, int n, basisFunction basisIn, basisFunction basisOut)
{
	pos.x += (repeater(make_float3(pos.x, pos.y, pos.z), scaleIn, seed, n, 2.0f, 0.5f, basisIn)) * strength;

	return repeater(pos, scaleOut, seed, n, 2.0f, 0.75f, basisOut);
}

__device__ float recursiveTurbulence(float3 pos, int n, float harmonic, float decay, float strength)
{
	float3 displace = make_float3(pos.x, pos.y, pos.z);
	float scale = 1.0f;
	float amp = 1.0f;
	float acc = 0.0f;

	for (int i = 0; i < n; i++)
	{
		acc += perlinNoise(scaleVector(displace, scale), 1.0f, 0) * amp;

		displace.x += perlinNoise(make_float3(pos.x, pos.y, pos.z), 1.0f, 0) * amp * strength;
		displace.y += acc * strength;
		displace.z += perlinNoise(make_float3(acc, acc, acc), 1.0f, 0) * amp * strength;

		scale *= harmonic;
		amp *= decay;
	}

	return acc / 1.0f;
}
