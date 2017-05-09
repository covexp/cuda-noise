// cudaNoise
// Common utility functions and typedefs

#ifndef util_cuh
#define util_cuh

namespace cudaNoise {

// Basis functions
typedef enum {
	BASIS_CHECKER,
	BASIS_DISCRETE,
	BASIS_LINEARVALUE,
	BASIS_FADEDVALUE,
	BASIS_CUBICVALUE,
	BASIS_PERLIN,
	BASIS_SIMPLEX,
	BASIS_WORLEY,
	BASIS_SPOTS
} basisFunction;

// Shaping functions
typedef enum {
	SHAPE_STEP,
	SHAPE_LINEAR,
	SHAPE_QUADRATIC
} profileShape;

// Function blending operators
typedef enum {
	OPERATOR_ADD,
	OPERATOR_AVG,
	OPERATOR_MUL,
	OPERATOR_MAX,
	OPERATOR_MIN
} repeatOperator;

#define EPSILON 0.000000001f

// Utility functions

// Hashing function (used for fast on-device pseudorandom numbers for randomness in noise)
__device__ __forceinline__ unsigned int hash(unsigned int seed)
{
	seed = (seed + 0x7ed55d16) + (seed << 12);
	seed = (seed ^ 0xc761c23c) ^ (seed >> 19);
	seed = (seed + 0x165667b1) + (seed << 5);
	seed = (seed + 0xd3a2646c) ^ (seed << 9);
	seed = (seed + 0xfd7046c5) + (seed << 3);
	seed = (seed ^ 0xb55a4f09) ^ (seed >> 16);

	return seed;
}

// Returns a random integer between [min, max]
__device__ __forceinline__ int randomIntRange(int min, int max, int seed)
{
	int base = hash(seed);
	base = base % (1 + max - min) + min;

	return base;
}

// Returns a random float between [0, 1]
__device__ __forceinline__ float randomFloat(unsigned int seed)
{
	unsigned int noiseVal = hash(seed);

	return ((float)noiseVal / (float)0xffffffff);
}

// Clamps val between [min, max]
__device__ __forceinline__ float clamp(float val, float min, float max)
{
	if (val < 0.0f)
		return 0.0f;
	else if (val > 1.0f)
		return 1.0f;

	return val;
}

// Maps from the signed range [0, 1] to unsigned [-1, 1]
// NOTE: no clamping
__device__ __forceinline__ float mapToSigned(float input)
{
	return input * 2.0f - 1.0f;
}

// Maps from the unsigned range [-1, 1] to signed [0, 1]
// NOTE: no clamping
__device__ __forceinline__ float mapToUnsigned(float input)
{
	return input * 0.5f + 0.5f;
}

// Maps from the signed range [0, 1] to unsigned [-1, 1] with clamping
__device__ __forceinline__ float clampToSigned(float input)
{
	return __saturatef(input) * 2.0f - 1.0f;
}

// Maps from the unsigned range [-1, 1] to signed [0, 1] with clamping
__device__ __forceinline__ float clampToUnsigned(float input)
{
	return __saturatef(input * 0.5f + 0.5f);
}


// Random float for a grid coordinate [-1, 1]
__device__ __forceinline__ float randomGrid(int x, int y, int z, int seed = 0)
{
	return mapToSigned(randomFloat((unsigned int)(x * 1723.0f + y * 93241.0f + z * 149812.0f + 3824.0f + seed)));
}

// Random unsigned int for a grid coordinate [0, MAXUINT]
__device__ __forceinline__ unsigned int randomIntGrid(int x, int y, int z, int seed = 0)
{
	return hash((unsigned int)(x * 1723.0f + y * 93241.0f + z * 149812.0f + 3824.0f + seed));
}

// Random 3D vector as float3 from grid position
__device__ __forceinline__ float3 vectorNoise(int x, int y, int z)
{
	return make_float3(randomFloat(x * 8231.0f + y * 34612.0f + z * 11836.0f + 19283.0f) * 2.0f - 1.0f,
		randomFloat(x * 1171.0f + y * 9234.0f + z * 992903.0f + 1466.0f) * 2.0f - 1.0f,
		0.0f);
}

// Scale 3D vector by scalar value
__device__ __forceinline__ float3 scaleVector(float3 v, float factor)
{
	return make_float3(v.x * factor, v.y * factor, v.z * factor);
}

// Adds two 3D vectors
__device__ __forceinline__ float3 addVectors(float3 v, float3 w)
{
	return make_float3(v.x + w.x, v.y + w.y, v.z + w.z);
}

// Dot product between two vectors
__device__ __forceinline__ float dotProduct(float3 u, float3 v)
{
	return (u.x * v.x + u.y * v.y + u.z * v.z);
}


}

#endif