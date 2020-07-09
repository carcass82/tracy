# Tracy [![Build Status](https://travis-ci.org/carcass82/tracy.svg?branch=master)](https://travis-ci.org/carcass82/tracy) [![Build Status](https://ci.appveyor.com/api/projects/status/rqsg04bl5sxoeigd?svg=true)](https://ci.appveyor.com/project/carcass82/tracy)


A simple raytracer based on *"Raytracing in One Weekend"* series by P. Shirley, you can find it [here](https://www.amazon.com/dp/B01B5AODD8) or [here](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

Mostly useful as profiling and optimization bench. I also used this project to start coding my own util functions / math library and to play with OpenMP and CUDA.

Tracing keep going as background task and image will keep getting better (with more samples) over time.
Camera can be moved in realtime using ``W``, ``A``, ``S``, ``D``, ``Q``, ``E`` keys and mouse (FPS-style).

Tracy only deals with triangle meshes (even builtin shapes like sphere or box are procedurally generated as triangle meshes) and uses a BVH/KdTree structure to speed up collision tests. ``data/scenes`` contains some simple and heavily commented scene description files. Scene file path can be specified as parameter. When started with no parameters Tracy tries to read ``data/default.scn``.

![teapot](doc/teapotscene.jpg)

![cornell](doc/cornell.jpg)

![dragon](doc/dragon.jpg)

### Building

Tracy uses CMake to handle different modules and dependencies. It defaults to the standard CPU raytracer with no external dependencies and (hopefully) sensible defaults.

![cmake](doc/cmake.jpg)
