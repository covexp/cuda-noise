# cudaNoise

*Library of common noise functions for CUDA kernels*

Device functions for use in CUDA kernels which provide tools for generating procedural 3D noise.
Basis noise functions can be combined for fractional Brownian motion, as well as used to perturb
the input vector for other noise functions for turbulence effects.

## Basis functions

![basis functions](http://covex.info/images/cudanoise_montage.jpg "Basis functions")

*Basis functions include: discrete noise, tricubic value noise, perlin gradient noise, simplex noise, spots and worley noise.*

## Derived functions

![repeater turbulence](http://covex.info/images/cudanoise.png "Repeater turbulence")

*Repeater turbulence of perlin noise functions.*

## Usage

cudaNoise comes as a single-header include library. Simply include cuda_noise.cuh in your CUDA source file, then call the cudaNoise functions from kernel or device functions. 

**NOTE: cudaNoise is not designed to be directly called from host code**

There is a simple texture viewer included in the /examples directory.

## Reference

### Basis functions

#### 3D Checker pattern

```cpp
float checker(float3 pos, float scale, int seed)
```

#### 3D Discrete noise

```cpp
float discreteNoise(float3 pos, float scale, int seed)
```

#### 3D Linear value noise

```cpp
float linearValue(float3 pos, float scale, int seed)
```

#### 3D Cubic value noise

```cpp
float cubicValue(float3 pos, float scale, int seed)
```

#### 3D Perlin gradient noise

```cpp
float perlinNoise(float3 pos, float scale, int seed)
```

#### 3D Simplex noise

```cpp
float simplexNoise(float3 pos, float scale, int seed)
```

#### 3D Worley cellular noise

```cpp
float worleyNoise(float3 pos, float scale, int seed, float size, int minNum, int maxNum, float jitter)
```

#### 3D Spots

```cpp
float spots(float3 pos, float scale, int seed, float size, int minNum, int maxNum, float jitter, profileShape shape)
```
