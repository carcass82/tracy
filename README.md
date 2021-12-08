# Tracy [![Build Status](https://travis-ci.org/carcass82/tracy.svg?branch=master)](https://app.travis-ci.com/github/carcass82/tracy) [![Build Status](https://ci.appveyor.com/api/projects/status/rqsg04bl5sxoeigd?svg=true)](https://ci.appveyor.com/project/carcass82/tracy)


A simple raytracer based on *"Raytracing in One Weekend"* series by P. Shirley, you can find it [here](https://www.amazon.com/dp/B01B5AODD8) or [here](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

Mostly useful as profiling and optimization bench. I also used this project to start coding my own util functions / math library and to play with OpenMP and CUDA.

Tracing keep going as background task and image will keep getting better (with more samples) over time.
Camera can be moved in realtime using ``W``, ``A``, ``S``, ``D``, ``Q``, ``E`` keys and mouse (FPS-style).

Tracy only deals with triangle meshes (even builtin shapes like sphere or box are procedurally generated as triangle meshes) and uses a BVH/KdTree structure to speed up collision tests. ``data/scenes`` contains some simple and heavily commented scene description files. Scene file path can be specified as parameter (``-scene``). When started with no parameters Tracy tries to read ``data/default.scn``.

Time for some screenshots:

![teapot](doc/teapotscene.jpg)

![cornell](doc/cornell.jpg)

![dragon](doc/dragon.jpg)

Tracy material system parametrization is [Unreal](https://docs.unrealengine.com/en-US/RenderingAndGraphics/Materials/PhysicallyBased/index.html)*-ish*, with *roughness* and *metalness* controlling metals and dielectrics while *translucent* and *IOR* parameters are used for translucent materials. Here's the usual spheres collection as example:

![materials](doc/materials.jpg)

  - 1st row shows *metal* spheres with roughness going from 0 to 1
  - in 2nd row spheres drop their metalness from 1 to 0 becoming a dielectric
  - 3rd row shows a dielectric material with its roughness growing from 0 to 1
  - 4th row contains translucent objects with a [*IOR*](https://en.wikipedia.org/wiki/List_of_refractive_indices) of 1.5 and roughness ranging from 0 to 1
  - last row materials have different *IORs*, from 1 to 2

Finally Tracy supports textured meshes and HDR sky probes, as shown below:

![textured](doc/textured.gif)

### Building

Tracy uses CMake to handle different modules and dependencies. It defaults to the standard CPU (both raytracer and raster) with no external dependencies and (hopefully) sensible defaults. Additional modules are enabled based on libraries found on system. Parameter ``-kernel`` selects module used for rendering.

![cmake](doc/cmake.jpg)
