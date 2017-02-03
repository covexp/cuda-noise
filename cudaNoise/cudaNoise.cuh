#ifndef cudanoise_cuh
#define cudanoise_cuh

// Utility functions
__device__ unsigned int hash(unsigned int a);
__device__ float getRandomValue(unsigned int seed);
__device__ float clamp(float val, float min, float max);
__device__ float mapToSigned(float input);
__device__ float mapToUnsigned(float input);
__device__ float whiteNoise(int x, int y, int z);

// Vector operations
__device__ float3 scaleVector(float3 v, float factor);
__device__ float3 addVectors(float3 v, float3 w);
__device__ float dotProduct(float3 u, float3 v);

// Helper functions for noise
__device__ float grad(int hash, float x, float y, float z);

// Noise functions
__device__ float checker(float x, float y, float z, float scale);
__device__ float discreteNoise(float x, float y, float z, float scale);
__device__ float perlinNoise(float3 pos);
__device__ float repeater(float3 pos, int n, float harmonic, float decay);
__device__ float turbulence(float3 pos, float strength);
__device__ float repeaterTurbulence(float3 pos, float strength, int n);
__device__ float recursiveTurbulence(float3 pos, int n, float harmonic, float decay, float strength);

__device__ float recursiveRepeaterTurbulence(float3 pos, int n, int m, float harmonic, float decay, float strength);

#endif