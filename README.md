# cudaNoise

*Library of common noise functions for CUDA kernels*

Device functions for use in CUDA kernels which provide tools for generating procedural 3D noise.
Basis noise functions can be combined for fractional Brownian motion, as well as used to perturb
the input vector for other noise functions for turbulence effects.

## Basis functions

![basis functions](http://covex.info/wp-content/uploads/2017/02/montage.jpg "Basis functions")

*Basis functions include: discrete noise, tricubic value noise, perlin gradient noise, simplex noise, spots and worley noise.*

## Derived functions

![repeater turbulence](http://covex.info/wp-content/uploads/2017/02/cudanoise-300x300.png "Repeater turbulence")

*Repeater turbulence of perlin noise functions.*

