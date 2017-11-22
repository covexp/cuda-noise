# cudaNoise

*Library of common noise functions for CUDA kernels*

Device functions for use in CUDA kernels which provide tools for generating procedural 3D noise.
Basis noise functions can be combined for fractional Brownian motion, as well as used to perturb
the input vector for other noise functions for turbulence effects.

## Basis functions

![basis functions](http://139.59.227.181/wp-content/uploads/2017/06/montage.jpg "Basis functions")

*Basis functions include: discrete noise, tricubic value noise, perlin gradient noise, simplex noise, spots and worley noise.*

## Derived functions

![repeater turbulence](http://139.59.227.181/wp-content/uploads/2017/06/cudanoise.png "Repeater turbulence")

*Repeater turbulence of perlin noise functions.*

