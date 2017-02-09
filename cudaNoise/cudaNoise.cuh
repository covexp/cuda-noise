#ifndef cudanoise_cuh
#define cudanoise_cuh

// Basis functions
typedef enum { CUDANOISE_CHECKER,
			   CUDANOISE_LINEARVALUE, 
			   CUDANOISE_FADEDVALUE,
			   CUDANOISE_CUBICVALUE,
			   CUDANOISE_PERLIN 
			 } basisFunction;

// Utility functions
__device__ unsigned int hash(unsigned int a);
__device__ float randomFloat(unsigned int seed);
__device__ float clamp(float val, float min, float max);
__device__ float mapToSigned(float input);
__device__ float mapToUnsigned(float input);
__device__ float randomGrid(int x, int y, int z, int seed);

// Vector operations
__device__ float3 scaleVector(float3 v, float factor);
__device__ float3 addVectors(float3 v, float3 w);
__device__ float dotProduct(float3 u, float3 v);

// Helper functions for noise
__device__ float grad(int hash, float x, float y, float z);
__device__ float cubic(float p0, float p1, float p2, float p3, float x);
__device__ float tricubic(int x, int y, int z, float u, float v, float w);

// Noise functions
__device__ float checker(float3 pos, float scale, int seed);
__device__ float discreteNoise(float3 pos, float scale, int seed);
__device__ float cubicValue(float3 pos, float scale, int seed);
__device__ float linearValue(float3 pos, float scale, int seed);
__device__ float fadedValue(float3 pos, float scale, int seed);
__device__ float perlinNoise(float3 pos, float scale, int seed);
__device__ float repeater(float3 pos, float scale, int seed, int n, float harmonic, float decay, basisFunction basis);
__device__ float turbulence(float3 pos, float scaleIn, float scaleOut, int seed, float strength, basisFunction inFunc, basisFunction outFunc);
__device__ float repeaterTurbulence(float3 pos, float scaleIn, float scaleOut, int seed, float strength, int n, basisFunction basisIn, basisFunction basisOut);
__device__ float recursiveTurbulence(float3 pos, int n, float harmonic, float decay, float strength);

#endif