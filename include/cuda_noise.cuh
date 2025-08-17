// cudaNoise
// Library of common 3D noise functions for CUDA kernels

#pragma once

#include <cuda_runtime.h>

namespace cudaNoise
{
struct BasisChecker;
struct BasisCubicValue;
struct BasisDiscrete;
struct BasisFadedValue;
struct BasisLinearValue;
struct BasisPerlin;
struct BasisSimplex;
struct BasisSpots;
struct BasisWorley;

namespace Basis
{
    struct Checker
    {
    };
    struct Discrete
    {
    };
    struct LinearValue
    {
    };
    struct FadedValue
    {
    };
    struct CubicValue
    {
    };
    struct Perlin
    {
    };
    struct Simplex
    {
    };
    struct Worley
    {
    };
    struct Spots
    {
    };
}

// Shaping functions
typedef enum
{
    SHAPE_STEP,
    SHAPE_LINEAR,
    SHAPE_QUADRATIC
} profileShape;

// Function blending operators
typedef enum
{
    OPERATOR_ADD,
    OPERATOR_AVG,
    OPERATOR_MUL,
    OPERATOR_MAX,
    OPERATOR_MIN
} repeatOperator;

constexpr float EPSILON = 0.000000001f;

// Utility functions

// Hashing function (used for fast on-device pseudorandom numbers for randomness in noise)
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

// Returns a random integer between [min, max]
__device__ int randomIntRange(int min, int max, int seed)
{
    int base = hash(seed);
    base = base % (1 + max - min) + min;

    return base;
}

// Returns a random float between [0, 1]
__device__ float randomFloat(unsigned int seed)
{
    unsigned int noiseVal = hash(seed);

    return ((float)noiseVal / (float)0xffffffff);
}

__device__ float3 fastRandomFloat3(unsigned int seed)
{
    unsigned int noiseVal = hash(seed);

    float x = (float)(noiseVal & 0x00000fff) / 4096.0f;
    float y = (float)(noiseVal & 0x00fff000 >> 12) / 4096.0f;
    float z = (float)(noiseVal >> 24) / 256.0f;

    return make_float3(x, y, z);
}

__device__ float4 fastRandomFloat4(unsigned int seed)
{
    unsigned int noiseVal = hash(seed);

    float x = (float)(noiseVal & 0xff) / 256.0f;
    float y = (float)(noiseVal >> 8 & 0xff) / 256.0f;
    float z = (float)(noiseVal >> 16 & 0xff) / 256.0f;
    float w = (float)(noiseVal >> 24) / 256.0f;

    return make_float4(x, y, z, w);
}

// Clamps val between [min, max]
__device__ float clamp(float val, float min, float max)
{
    if (val < min)
        return min;
    else if (val > max)
        return max;

    return val;
}

// Maps from the signed range [0, 1] to unsigned [-1, 1]
// NOTE: no clamping
__device__ float mapToSigned(float input)
{
    return input * 2.0f - 1.0f;
}

// Maps from the unsigned range [-1, 1] to signed [0, 1]
// NOTE: no clamping
__device__ float mapToUnsigned(float input)
{
    return input * 0.5f + 0.5f;
}

// Maps from the signed range [0, 1] to unsigned [-1, 1] with clamping
__device__ float clampToSigned(float input)
{
    return __saturatef(input) * 2.0f - 1.0f;
}

// Maps from the unsigned range [-1, 1] to signed [0, 1] with clamping
__device__ float clampToUnsigned(float input)
{
    return __saturatef(input * 0.5f + 0.5f);
}


// Random float for a grid coordinate [-1, 1]
__device__ float randomGrid(int x, int y, int z, int seed = 0)
{
    return mapToSigned(randomFloat((unsigned int)(x * 1723.0f + y * 93241.0f + z * 149812.0f + 3824.0f + seed)));
}

// Random unsigned int for a 3D grid coordinate [0, MAXUINT]
__device__ unsigned int randomIntGrid(float x, float y, float z, float seed)
{
    return hash((unsigned int)(x * 1723.0f + y * 93241.0f + z * 149812.0f + 3824 + seed));
}

// Random unsigned int for a 4D grid coordinate [0, MAXUINT]
__device__ unsigned int randomIntGrid(float x, float y, float z, float w, float seed)
{
    return hash((unsigned int)(x * 1723.0f + y * 93241.0f + z * 149812.0f + w * 61838.0f + 3824 + seed));
}

// Random 3D vector as float3 from grid position
__device__ float3 vectorNoise(int x, int y, int z)
{
    return make_float3(randomFloat(x * 8231.0f + y * 34612.0f + z * 11836.0f + 19283.0f) * 2.0f - 1.0f,
                       randomFloat(x * 1171.0f + y * 9234.0f + z * 992903.0f + 1466.0f) * 2.0f - 1.0f,
                       0.0f);
}

// Scale 3D vector by scalar value
__device__ float3 scaleVector(float3 v, float factor)
{
    return make_float3(v.x * factor, v.y * factor, v.z * factor);
}

// Scale 3D vector by nonuniform parameters
__device__ float3 nonuniformScaleVector(float3 v, float xf, float yf, float zf)
{
    return make_float3(v.x * xf, v.y * yf, v.z * zf);
}


// Adds two 3D vectors
__device__ float3 addVectors(float3 v, float3 w)
{
    return make_float3(v.x + w.x, v.y + w.y, v.z + w.z);
}

// Dot product between two vectors
__device__ float dotProduct(float3 u, float3 v)
{
    return (u.x * v.x + u.y * v.y + u.z * v.z);
}

// Device constants for noise

__device__ __constant__ float gradMap[16][3] = {{1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {-1.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 1.0f}, {-1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 1.0f}, {0.0f, -1.0f, 1.0f}, {0.0f, 1.0f, -1.0f}, {0.0f, -1.0f, -1.0f}};

// Helper functions for noise

// Linearly interpolate between two float values
__device__ float lerp(float a, float b, float ratio)
{
    return a * (1.0f - ratio) + b * ratio;
}

// 1D cubic interpolation with four points
__device__ float cubic(float p0, float p1, float p2, float p3, float x)
{
    return p1 + 0.5f * x * (p2 - p0 + x * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + x * (3.0f * (p1 - p2) + p3 - p0)));
}

// Fast gradient function for gradient noise
__device__ float grad(int hash, float x, float y, float z)
{
    switch (hash & 0xF)
        {
            case 0x0:
                return x + y;
            case 0x1:
                return -x + y;
            case 0x2:
                return x - y;
            case 0x3:
                return -x - y;
            case 0x4:
                return x + z;
            case 0x5:
                return -x + z;
            case 0x6:
                return x - z;
            case 0x7:
                return -x - z;
            case 0x8:
                return y + z;
            case 0x9:
                return -y + z;
            case 0xA:
                return y - z;
            case 0xB:
                return -y - z;
            case 0xC:
                return y + x;
            case 0xD:
                return -y + z;
            case 0xE:
                return y - x;
            case 0xF:
                return -y - z;
            default:
                return 0; // never happens
        }
}

__device__ float grad(int hash, float x, float y, float z, float w)
{
    int h = hash & 31;
    double u, v;

    if (h < 24)
        {
            if (h < 16)
                {
                    u = x;
                    v = y;
                }
            else
                {
                    u = z;
                    v = w;
                }
        }
    else
        {
            if (h < 28)
                {
                    u = y;
                    v = z;
                }
            else
                {
                    u = w;
                    v = x;
                }
        }

    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}


// Ken Perlin's fade function for Perlin noise
__device__ float fade(float t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); // 6t^5 - 15t^4 + 10t^3
}

// Dot product using a float[3] and float parameters
// NOTE: could be cleaned up
__device__ float dot(float g[3], float x, float y, float z)
{
    return g[0] * x + g[1] * y + g[2] * z;
}

// Random value for simplex noise [0, 255]
__device__ unsigned char calcPerm(int p)
{
    return (unsigned char)(hash(p));
}

// Random value for simplex noise [0, 11]
__device__ unsigned char calcPerm12(int p)
{
    return (unsigned char)(hash(p) % 12);
}

// Noise functions

// Simplex noise adapted from Java code by Stefan Gustafson and Peter Eastman
__device__ float simplexNoise(float3 pos, float scale, int seed)
{
    float xin = pos.x * scale;
    float yin = pos.y * scale;
    float zin = pos.z * scale;

    // Skewing and unskewing factors for 3 dimensions
    float F3 = 1.0f / 3.0f;
    float G3 = 1.0f / 6.0f;

    float n0, n1, n2, n3; // Noise contributions from the four corners

    // Skew the input space to determine which simplex cell we're in
    float s = (xin + yin + zin) * F3; // Very nice and simple skew factor for 3D
    int i = floorf(xin + s);
    int j = floorf(yin + s);
    int k = floorf(zin + s);
    float t = (i + j + k) * G3;
    float X0 = i - t; // Unskew the cell origin back to (x,y,z) space
    float Y0 = j - t;
    float Z0 = k - t;
    float x0 = xin - X0; // The x,y,z distances from the cell origin
    float y0 = yin - Y0;
    float z0 = zin - Z0;

    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    // Determine which simplex we are in.
    int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
    int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
    if (x0 >= y0)
        {
            if (y0 >= z0)
                {
                    i1 = 1.0f;
                    j1 = 0.0f;
                    k1 = 0.0f;
                    i2 = 1.0f;
                    j2 = 1.0f;
                    k2 = 0.0f;
                } // X Y Z order
            else if (x0 >= z0)
                {
                    i1 = 1.0f;
                    j1 = 0.0f;
                    k1 = 0.0f;
                    i2 = 1.0f;
                    j2 = 0.0f;
                    k2 = 1.0f;
                } // X Z Y order
            else
                {
                    i1 = 0.0f;
                    j1 = 0.0f;
                    k1 = 1.0f;
                    i2 = 1.0f;
                    j2 = 0.0f;
                    k2 = 1.0f;
                } // Z X Y order
        }
    else
        { // x0<y0
            if (y0 < z0)
                {
                    i1 = 0.0f;
                    j1 = 0.0f;
                    k1 = 1.0f;
                    i2 = 0.0f;
                    j2 = 1;
                    k2 = 1.0f;
                } // Z Y X order
            else if (x0 < z0)
                {
                    i1 = 0.0f;
                    j1 = 1.0f;
                    k1 = 0.0f;
                    i2 = 0.0f;
                    j2 = 1.0f;
                    k2 = 1.0f;
                } // Y Z X order
            else
                {
                    i1 = 0.0f;
                    j1 = 1.0f;
                    k1 = 0.0f;
                    i2 = 1.0f;
                    j2 = 1.0f;
                    k2 = 0.0f;
                } // Y X Z order
        }

    // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
    // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
    // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
    // c = 1/6.
    float x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
    float y1 = y0 - j1 + G3;
    float z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2.0f * G3; // Offsets for third corner in (x,y,z) coords
    float y2 = y0 - j2 + 2.0f * G3;
    float z2 = z0 - k2 + 2.0f * G3;
    float x3 = x0 - 1.0f + 3.0f * G3; // Offsets for last corner in (x,y,z) coords
    float y3 = y0 - 1.0f + 3.0f * G3;
    float z3 = z0 - 1.0f + 3.0f * G3;

    int gi0 = calcPerm12(seed + (i * 607495) + (j * 359609) + (k * 654846));
    int gi1 = calcPerm12(seed + (i + i1) * 607495 + (j + j1) * 359609 + (k + k1) * 654846);
    int gi2 = calcPerm12(seed + (i + i2) * 607495 + (j + j2) * 359609 + (k + k2) * 654846);
    int gi3 = calcPerm12(seed + (i + 1) * 607495 + (j + 1) * 359609 + (k + 1) * 654846);

    // Calculate the contribution from the four corners
    float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
    if (t0 < 0.0f)
        n0 = 0.0f;
    else
        {
            t0 *= t0;
            n0 = t0 * t0 * dot(gradMap[gi0], x0, y0, z0);
        }
    float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
    if (t1 < 0.0f)
        n1 = 0.0f;
    else
        {
            t1 *= t1;
            n1 = t1 * t1 * dot(gradMap[gi1], x1, y1, z1);
        }
    float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
    if (t2 < 0.0f)
        n2 = 0.0f;
    else
        {
            t2 *= t2;
            n2 = t2 * t2 * dot(gradMap[gi2], x2, y2, z2);
        }
    float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
    if (t3 < 0.0f)
        n3 = 0.0f;
    else
        {
            t3 *= t3;
            n3 = t3 * t3 * dot(gradMap[gi3], x3, y3, z3);
        }

    // Add contributions from each corner to get the final noise value.
    // The result is scaled to stay just inside [-1,1]
    return 32.0f * (n0 + n1 + n2 + n3);
}

// Checker pattern
__device__ float checker(float3 pos, float scale, int seed)
{
    int ix = (int)(pos.x * scale);
    int iy = (int)(pos.y * scale);
    int iz = (int)(pos.z * scale);

    if ((ix + iy + iz) % 2 == 0)
    {
            return 1.0f;
        }

    return -1.0f;
}

// Random spots
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
                            int numSpots = randomIntRange(minNum, maxNum, seed + (ix + x) * 823746.0f + (iy + y) * 12306.0f + (iz + z) * 823452.0f + 3234874.0f);

                            for (int i = 0; i < numSpots; i++)
                                {
                                    float distU = u - x - (randomFloat(seed + (ix + x) * 23784.0f + (iy + y) * 9183.0f + (iz + z) * 23874.0f * i + 27432.0f) * jitter - jitter / 2.0f);
                                    float distV = v - y - (randomFloat(seed + (ix + x) * 12743.0f + (iy + y) * 45191.0f + (iz + z) * 144421.0f * i + 76671.0f) * jitter - jitter / 2.0f);
                                    float distW = w - z - (randomFloat(seed + (ix + x) * 82734.0f + (iy + y) * 900213.0f + (iz + z) * 443241.0f * i + 199823.0f) * jitter - jitter / 2.0f);

                                    float distanceSq = distU * distU + distV * distV + distW * distW;
                                    float distanceAbs = 0.0f;

                                    switch (shape)
                                        {
                                            case (SHAPE_STEP):
                                                if (distanceSq < size)
                                                    val = fmaxf(val, 1.0f);
                                                else
                                                    val = fmaxf(val, -1.0f);
                                                break;
                                            case (SHAPE_LINEAR):
                                                distanceAbs = fabsf(distU) + fabsf(distV) + fabsf(distW);
                                                val = fmaxf(val, 1.0f - clamp(distanceAbs, 0.0f, size) / size);
                                                break;
                                            case (SHAPE_QUADRATIC):
                                                val = fmaxf(val, 1.0f - clamp(distanceSq, 0.0f, size) / size);
                                                break;
                                        }
                                }
                        }
                }
        }

    return val;
}


// Worley cellular noise
__device__ float worleyNoise(float3 pos, float scale, int seed, float size,
                             int minNum, int maxNum, float jitter)
{
    if (size < EPSILON)
        return 0.0f;

    int ix = (int)(pos.x * scale);
    int iy = (int)(pos.y * scale);
    int iz = (int)(pos.z * scale);

    float u = pos.x - (float)ix;
    float v = pos.y - (float)iy;
    float w = pos.z - (float)iz;

    float minDist = 1000000.0f;

    // Traverse the whole 3x3 neighborhood looking for the closest feature point
    for (int x = -2; x < 3; x++)
        {
            for (int y = -2; y < 3; y++)
                {
                    for (int z = -2; z < 3; z++)
                        {
                            int numPoints =
                                randomIntRange(minNum, maxNum,
                                               seed + (ix + x) * 823746.0f + (iy + y) * 12306.0f +
                                                   (iz + z) * 67262.0f);

                            for (int i = 0; i < numPoints; i++)
                                {
                                    float distU =
                                        u - x -
                                        (randomFloat(seed + (ix + x) * 23784.0f + (iy + y) * 9183.0f +
                                                     (iz + z) * 23874.0f * i + 27432.0f) *
                                             jitter -
                                         jitter / 2.0f);
                                    float distV =
                                        v - y -
                                        (randomFloat(seed + (ix + x) * 12743.0f + (iy + y) * 45191.0f +
                                                     (iz + z) * 144421.0f * i + 76671.0f) *
                                             jitter -
                                         jitter / 2.0f);
                                    float distW =
                                        w - z -
                                        (randomFloat(seed + (ix + x) * 82734.0f + (iy + y) * 900213.0f +
                                                     (iz + z) * 443241.0f * i + 199823.0f) *
                                             jitter -
                                         jitter / 2.0f);

                                    float distanceSq = distU * distU + distV * distV + distW * distW;

                                    if (distanceSq < minDist)
                                        minDist = distanceSq;
                                }
                        }
                }
        }

    return __saturatef(minDist) * 2.0f - 1.0f;
}

__device__ float worley333(const float3 &pos, int seed, int minNum, int maxNum, float jitter = 1.0f)
{
    using namespace cudaNoise;

    const int ix = (int)pos.x;
    const int iy = (int)pos.y;
    const int iz = (int)pos.z;

    const float u = pos.x - (float)ix;
    const float v = pos.y - (float)iy;
    const float w = pos.z - (float)iz;

    float minDist = 1000000.0f;

    // Traverse the whole 3x3 neighborhood looking for the closest feature point
    for (int x = -1; x < 2; ++x)
        {
            unsigned int x_seed = ((ix + x + 47408771) * 49085063) ^ seed;
            for (int y = -1; y < 2; ++y)
                {
                    unsigned int y_seed = ((iy + y + 41551667) * 46698571) ^ x_seed;
                    for (int z = -1; z < 2; ++z)
                        {
                            unsigned int point_seed = ((iz + z + 16687919) * 53722831) ^ y_seed;
                            const int numPoints =
                                randomIntRange(minNum, maxNum, point_seed ^ 65190661);

                            for (int i = 0; i < numPoints; ++i)
                                {
                                    unsigned int i_seed = ((i + 23535101) * 23308841) ^ point_seed;
                                    const float distU =
                                        u - static_cast<float>(x) - 0.5f -
                                        (randomFloat(i_seed ^ 47047237) * jitter - jitter / 2.0f);
                                    const float distV =
                                        v - static_cast<float>(y) - 0.5f -
                                        (randomFloat(i_seed ^ 46759411) * jitter - jitter / 2.0f);
                                    const float distW =
                                        w - static_cast<float>(z) - 0.5f -
                                        (randomFloat(i_seed ^ 90628337) * jitter - jitter / 2.0f);

                                    const float distanceSq =
                                        distU * distU + distV * distV + distW * distW;

                                    if (distanceSq < minDist)
                                        {
                                            minDist = distanceSq;
                                        }
                                }
                        }
                }
        }

    return __saturatef(minDist);
}

__device__ float worleySingle(const float3 &pos, int seed)
{
    static constexpr float jitter = 1.0f;

    const int ix = (int)pos.x;
    const int iy = (int)pos.y;
    const int iz = (int)pos.z;

    const float u = pos.x - (float)ix;
    const float v = pos.y - (float)iy;
    const float w = pos.z - (float)iz;

    float minDist = 1000000.0f;

    // Traverse the whole 3x3 neighborhood looking for the closest feature point
    for (int x = -1; x < 2; ++x)
        {
            unsigned int x_seed = ((ix + x + 47408771) * 49085063) ^ seed;
            for (int y = -1; y < 2; ++y)
                {
                    unsigned int xy_seed = ((iy + y + 41551667) * 46698571) ^ x_seed;
                    for (int z = -1; z < 2; ++z)
                        {
                            unsigned int xyz_seed = ((iz + z + 16687919) * 53722831) ^ xy_seed;
                            const float distU =
                                u - static_cast<float>(x) - 0.5f -
                                (randomFloat(xyz_seed ^ 47047237) * jitter - jitter / 2.0f);
                            const float distV =
                                v - static_cast<float>(y) - 0.5f -
                                (randomFloat(xyz_seed ^ 46759411) * jitter - jitter / 2.0f);
                            const float distW =
                                w - static_cast<float>(z) - 0.5f -
                                (randomFloat(xyz_seed ^ 90628337) * jitter - jitter / 2.0f);

                            float distanceSq =
                                distU * distU + distV * distV + distW * distW;

                            if (distanceSq < minDist)
                                {
                                    minDist = distanceSq;
                                }
                        }
                }
        }

    return __saturatef(minDist);
}

__device__ float worleyVarying(const float3 &pos, int seed)
{
    static constexpr float jitter = 1.0f;

    const int ix = (int)pos.x;
    const int iy = (int)pos.y;
    const int iz = (int)pos.z;

    const float u = pos.x - (float)ix;
    const float v = pos.y - (float)iy;
    const float w = pos.z - (float)iz;

    float minDist = 1000000.0f;
    float brightness = 1.0f;

    // Traverse the whole 3x3 neighborhood looking for the closest feature point
    for (int x = -1; x < 2; ++x)
        {
            unsigned int x_seed = ((ix + x + 47408771) * 49085063) ^ seed;
            for (int y = -1; y < 2; ++y)
                {
                    unsigned int xy_seed = ((iy + y + 41551667) * 46698571) ^ x_seed;
                    for (int z = -1; z < 2; ++z)
                        {
                            unsigned int xyz_seed = ((iz + z + 16687919) * 53722831) ^ xy_seed;
                            const float distU =
                                u - static_cast<float>(x) - 0.5f -
                                (randomFloat(xyz_seed ^ 47047237) * jitter - jitter / 2.0f);
                            const float distV =
                                v - static_cast<float>(y) - 0.5f -
                                (randomFloat(xyz_seed ^ 46759411) * jitter - jitter / 2.0f);
                            const float distW =
                                w - static_cast<float>(z) - 0.5f -
                                (randomFloat(xyz_seed ^ 90628337) * jitter - jitter / 2.0f);

                            float distanceSq =
                                distU * distU + distV * distV + distW * distW;

                            if (distanceSq < minDist)
                                {
                                    brightness = randomFloat(xyz_seed ^ 7239484);
                                    minDist = distanceSq;
                                }
                        }
                }
        }

    return __saturatef((1.0f - minDist) * brightness);
}


__device__ float worleySkip(const float3 &pos, int seed, int minNum,
                            int maxNum)
{
    using namespace cudaNoise;

    static constexpr float jitter = 0.5f;

    const int ix = (int)pos.x;
    const int iy = (int)pos.y;
    const int iz = (int)pos.z;

    const float u = pos.x - (float)ix;
    const float v = pos.y - (float)iy;
    const float w = pos.z - (float)iz;

    float minDist = 1000000.0f;

    // Traverse the whole 3x3 neighborhood looking for the closest feature point
    for (int x = -1; x < 2; ++x)
        {
            if (u > 0.5f && x == -1)
                continue;
            if (u < 0.5f && x == 1)
                continue;

            unsigned int x_seed = ((ix + x + 47408771) * 49085063) ^ seed;
            for (int y = -1; y < 2; ++y)
                {
                    if (v > 0.5f && y == -1)
                        continue;
                    if (v < 0.5f && y == 1)
                        continue;

                    unsigned int y_seed = ((iy + y + 41551667) * 46698571) ^ x_seed;
                    for (int z = -1; z < 2; ++z)
                        {
                            if (w > 0.5f && z == -1)
                                continue;
                            if (w < 0.5f && z == 1)
                                continue;

                            unsigned int point_seed = ((iz + z + 16687919) * 53722831) ^ y_seed;
                            const int numPoints =
                                randomIntRange(minNum, maxNum, point_seed ^ 65190661);

                            for (int i = 0; i < numPoints; ++i)
                                {
                                    unsigned int i_seed = ((i + 23535101) * 23308841) ^ point_seed;
                                    const float distU =
                                        u - static_cast<float>(x) - jitter -
                                        (randomFloat(i_seed ^ 47047237) * jitter - jitter / 2.0f);
                                    const float distV =
                                        v - static_cast<float>(y) - jitter -
                                        (randomFloat(i_seed ^ 46759411) * jitter - jitter / 2.0f);
                                    const float distW =
                                        w - static_cast<float>(z) - jitter -
                                        (randomFloat(i_seed ^ 90628337) * jitter - jitter / 2.0f);

                                    const float distanceSq =
                                        distU * distU + distV * distV + distW * distW;

                                    if (distanceSq < minDist)
                                        {
                                            minDist = distanceSq;
                                        }
                                }
                        }
                }
        }

    return __saturatef(minDist * 2.0f);
}

// Worley cellular noise
__device__ float worleyVarientNoise(float3 pos, float scale, unsigned int seed, int minPoints, int maxPoints, float variation, float jitter)
{
    int ix = (int)(pos.x * scale);
    int iy = (int)(pos.y * scale);
    int iz = (int)(pos.z * scale);

    float u = pos.x - (float)ix;
    float v = pos.y - (float)iy;
    float w = pos.z - (float)iz;

    float minDist = 1000000.0f;

    // Traverse the whole 3x3 neighborhood looking for the closest feature point
    for (int x = -1; x < 2; x++)
        {
            for (int y = -1; y < 2; y++)
                {
                    for (int z = -1; z < 2; z++)
                        {
                            //                int numPoints = randomIntRange(minPoints, maxPoints, seed ^ ((ix + x + 723842u) * 872342u) ^ ((iy + y + 183724u) * 823742u) ^ ((iz + z + 423528u) * 12657u));
                            int numPoints = 8;

                            for (int i = 0; i < numPoints; i++)
                                {
                                    float4 rnd4 = fastRandomFloat4(seed ^ ((ix + x + 273282u) * 1934578u) ^ ((iy + y + 523482u) * 972223u) ^ ((iz + z + 2872345u) * 7224356u));
                                    float distU = u - x - (rnd4.x * jitter - jitter / 2.0f);
                                    float distV = v - y - (rnd4.y * jitter - jitter / 2.0f);
                                    float distW = w - z - (rnd4.z * jitter - jitter / 2.0f);

                                    float distanceSq = distU * distU + distV * distV + distW * distW;

                                    minDist = fminf(minDist, distanceSq * (rnd4.w + 0.5f));
                                }
                        }
                }
        }

    return __saturatef(minDist) * 2.0f - 1.0f;
}

// Tricubic interpolation
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

// Discrete noise (nearest neighbor)
__device__ float discreteNoise(float3 pos, float scale, int seed)
{
    int ix = (int)(pos.x * scale);
    int iy = (int)(pos.y * scale);
    int iz = (int)(pos.z * scale);

    return randomGrid(ix, iy, iz, seed);
}

// Linear value noise
__device__ float linearValue(float3 pos, float scale, int seed)
{
    float fseed = (float)seed;

    int ix = (int)pos.x;
    int iy = (int)pos.y;
    int iz = (int)pos.z;

    float u = pos.x - ix;
    float v = pos.y - iy;
    float w = pos.z - iz;

    // Corner values
    float a000 = randomGrid(ix, iy, iz, fseed);
    float a100 = randomGrid(ix + 1, iy, iz, fseed);
    float a010 = randomGrid(ix, iy + 1, iz, fseed);
    float a110 = randomGrid(ix + 1, iy + 1, iz, fseed);
    float a001 = randomGrid(ix, iy, iz + 1, fseed);
    float a101 = randomGrid(ix + 1, iy, iz + 1, fseed);
    float a011 = randomGrid(ix, iy + 1, iz + 1, fseed);
    float a111 = randomGrid(ix + 1, iy + 1, iz + 1, fseed);

    // Linear interpolation
    float x00 = lerp(a000, a100, u);
    float x10 = lerp(a010, a110, u);
    float x01 = lerp(a001, a101, u);
    float x11 = lerp(a011, a111, u);

    float y0 = lerp(x00, x10, v);
    float y1 = lerp(x01, x11, v);

    return lerp(y0, y1, w);
}

// Linear value noise smoothed with Perlin's fade function
__device__ float fadedValue(float3 pos, float scale, int seed)
{
    float fseed = (float)seed;

    int ix = (int)(pos.x * scale);
    int iy = (int)(pos.y * scale);
    int iz = (int)(pos.z * scale);

    float u = fade(pos.x - ix);
    float v = fade(pos.y - iy);
    float w = fade(pos.z - iz);

    // Corner values
    float a000 = randomGrid(ix, iy, iz, fseed);
    float a100 = randomGrid(ix + 1, iy, iz, fseed);
    float a010 = randomGrid(ix, iy + 1, iz, fseed);
    float a110 = randomGrid(ix + 1, iy + 1, iz, fseed);
    float a001 = randomGrid(ix, iy, iz + 1, fseed);
    float a101 = randomGrid(ix + 1, iy, iz + 1, fseed);
    float a011 = randomGrid(ix, iy + 1, iz + 1, fseed);
    float a111 = randomGrid(ix + 1, iy + 1, iz + 1, fseed);

    // Linear interpolation
    float x00 = lerp(a000, a100, u);
    float x10 = lerp(a010, a110, u);
    float x01 = lerp(a001, a101, u);
    float x11 = lerp(a011, a111, u);

    float y0 = lerp(x00, x10, v);
    float y1 = lerp(x01, x11, v);

    return lerp(y0, y1, w) / 2.0f * 1.0f;
}

// Tricubic interpolated value noise
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

// Perlin gradient noise 3D
__device__ float perlinNoise(float3 pos, float scale, int seed)
{
    float fseed = (float)seed;

    pos.x = pos.x * scale;
    pos.y = pos.y * scale;
    pos.z = pos.z * scale;

    // zero corner integer position
    float ix = floorf(pos.x);
    float iy = floorf(pos.y);
    float iz = floorf(pos.z);

    // current position within unit cube
    pos.x -= ix;
    pos.y -= iy;
    pos.z -= iz;

    // adjust for fade
    float u = fade(pos.x);
    float v = fade(pos.y);
    float w = fade(pos.z);

    // influence values
    float i000 = grad(randomIntGrid(ix, iy, iz, fseed), pos.x, pos.y, pos.z);
    float i100 = grad(randomIntGrid(ix + 1.0f, iy, iz, fseed), pos.x - 1.0f, pos.y, pos.z);
    float i010 = grad(randomIntGrid(ix, iy + 1.0f, iz, fseed), pos.x, pos.y - 1.0f, pos.z);
    float i110 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz, fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z);
    float i001 = grad(randomIntGrid(ix, iy, iz + 1.0f, fseed), pos.x, pos.y, pos.z - 1.0f);
    float i101 = grad(randomIntGrid(ix + 1.0f, iy, iz + 1.0f, fseed), pos.x - 1.0f, pos.y, pos.z - 1.0f);
    float i011 = grad(randomIntGrid(ix, iy + 1.0f, iz + 1.0f, fseed), pos.x, pos.y - 1.0f, pos.z - 1.0f);
    float i111 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz + 1.0f, fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z - 1.0f);

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

// Perlin gradient noise 4D
__device__ float perlinNoise(float4 pos, float scale, int seed)
{
    float fseed = (float)seed;

    pos.x = pos.x * scale;
    pos.y = pos.y * scale;
    pos.z = pos.z * scale;
    pos.w = pos.w * scale;

    // zero corner integer position
    float ix = floorf(pos.x);
    float iy = floorf(pos.y);
    float iz = floorf(pos.z);
    float iw = floorf(pos.w);

    // current position within unit hypercube
    pos.x -= ix;
    pos.y -= iy;
    pos.z -= iz;
    pos.w -= iw;

    // adjust for fade
    float u = fade(pos.x);
    float v = fade(pos.y);
    float w = fade(pos.z);
    float t = fade(pos.w);

    // influence values
    float i0000 = grad(randomIntGrid(ix, iy, iz, iw, fseed), pos.x, pos.y, pos.z, pos.w);
    float i1000 = grad(randomIntGrid(ix + 1.0f, iy, iz, iw, fseed), pos.x - 1.0f, pos.y, pos.z, pos.w);
    float i0100 = grad(randomIntGrid(ix, iy + 1.0f, iz, iw, fseed), pos.x, pos.y - 1.0f, pos.z, pos.w);
    float i1100 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz, iw, fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z, pos.w);
    float i0010 = grad(randomIntGrid(ix, iy, iz + 1.0f, iw, fseed), pos.x, pos.y, pos.z - 1.0f, pos.w);
    float i1010 = grad(randomIntGrid(ix + 1.0f, iy, iz + 1.0f, iw, fseed), pos.x - 1.0f, pos.y, pos.z - 1.0f, pos.w);
    float i0110 = grad(randomIntGrid(ix, iy + 1.0f, iz + 1.0f, iw, fseed), pos.x, pos.y - 1.0f, pos.z - 1.0f, pos.w);
    float i1110 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz + 1.0f, iw, fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z - 1.0f, pos.w);
    float i0001 = grad(randomIntGrid(ix, iy, iz, iw + 1.0f, fseed), pos.x, pos.y, pos.z, pos.w - 1.0f);
    float i1001 = grad(randomIntGrid(ix + 1.0f, iy, iz, iw + 1.0f, fseed), pos.x - 1.0f, pos.y, pos.z, pos.w - 1.0f);
    float i0101 = grad(randomIntGrid(ix, iy + 1.0f, iz, iw + 1.0f, fseed), pos.x, pos.y - 1.0f, pos.z, pos.w - 1.0f);
    float i1101 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz, iw + 1.0f, fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z, pos.w - 1.0f);
    float i0011 = grad(randomIntGrid(ix, iy, iz + 1.0f, iw + 1.0f, fseed), pos.x, pos.y, pos.z - 1.0f, pos.w - 1.0f);
    float i1011 = grad(randomIntGrid(ix + 1.0f, iy, iz + 1.0f, iw + 1.0f, fseed), pos.x - 1.0f, pos.y, pos.z - 1.0f, pos.w - 1.0f);
    float i0111 = grad(randomIntGrid(ix, iy + 1.0f, iz + 1.0f, iw + 1.0f, fseed), pos.x, pos.y - 1.0f, pos.z - 1.0f, pos.w - 1.0f);
    float i1111 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz + 1.0f, iw + 1.0f, fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z - 1.0f, pos.w - 1.0f);

    // interpolate values
    float x00 = lerp(i0000, i1000, u);
    float x10 = lerp(i0100, i1100, u);
    float x01 = lerp(i0010, i1010, u);
    float x11 = lerp(i0110, i1110, u);
    float y0 = lerp(x00, x10, v);
    float y1 = lerp(x01, x11, v);
    float z0 = lerp(y0, y1, w);

    x00 = lerp(i0001, i1001, u);
    x10 = lerp(i0101, i1101, u);
    x01 = lerp(i0011, i1011, u);
    x11 = lerp(i0111, i1111, u);
    y0 = lerp(x00, x10, v);
    y1 = lerp(x01, x11, v);
    float z1 = lerp(y0, y1, w);

    return lerp(z0, z1, t);
}

// Derived noise functions

// Sampling functions are defined as overloads of __sample() with the basis function marker as the first parameter

// Sample Perlin gradient noise
//template <typename ...Args>
//__device__ float __sample(Basis::Perlin basis, Args&&... args)
//{
//    return perlinNoise(args...);
//}

// Sample Simplex noise
template <typename BasisSimplex, typename... Args>
__device__ float __sample(Args&&... args)
{
    return simplexNoise(std::forward<Args>(args)...);
}

// Sample Spots noise
//template <typename... Args>
//__device__ float __sample(Basis::Spots basis, Args &&...args)
//{
//    return spots(args...);
//}
//
//// Sample Checker noise
//template <typename... Args>
//__device__ float __sample<Basis::Checker>(Args &&...args)
//{
//    return checker(args...);
//}
//
//// Sample Discrete noise
//template <typename... Args>
//__device__ float __sample(Basis::Discrete basis, Args &&...args)
//{
//    return discreteNoise(args...);
//}
//
//// Sample linear value noise
//template <typename... Args>
//__device__ float __sample(Basis::LinearValue basis, Args &&...args)
//{
//    return linearValue(args...);
//}
//
//// Sample faded value noise
//template <typename... Args>
//__device__ float __sample(Basis::FadedValue basis, Args &&...args)
//{
//    return fadedValue(args...);
//}
//
//// Sample cubic value noise
//template <typename... Args>
//__device__ float __sample(Basis::CubicValue basis, Args &&...args)
//{
//    return cubicValue(args...);
//}
//
//// Sample Worley noise
//template <typename... Args>
//__device__ float __sample(Basis::Worley basis, Args &&...args)
//{
//    return worleyNoise(args...);
//}

template <typename B>
__device__ float fbm(int n, float lacunarity, float decay, float3 pos, float scale, int seed)
{
    float acc = 0.0f;
    float amp = 1.0f;

    for (int i = 0; i < n; i++)
        {
            acc += __sample<B>(pos, scale, seed) * amp * 0.5f;
            scale *= lacunarity;
            amp *= decay;
            seed = seed ^ ((i + 198273) * 928374);
        }

    return acc;
}

template <typename B>
__device__ float fbmAbs(int n, float lacunarity, float decay, float3 pos, float scale, int seed)
{
    float acc = 0.0f;
    float amp = 1.0f;

    for (int i = 0; i < n; i++)
        {
            acc += fabsf(__sample<B>(pos, scale, seed)) * amp * 0.5f;
            scale *= lacunarity;
            amp *= decay;
            seed = seed ^ ((i + 198273) * 928374);
        }

    return mapToSigned(acc);
}

} // namespace cudaNoise
