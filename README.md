# Tracy #

Your friendly kawaii raytracer (more a path tracer, actually).

### Compiling ###

* needs glm (who doesn't?)

`g++/clang++ -oraytracer raytracer.cpp -O3 -fopenmp -ffast-math -march=native -std=c++11`

or

`cl.exe /nologo /EHsc- /fp:fast /Ox /openmp /favor:INTEL64 raytracer.cpp`

*(flags for best performance evah on my machine)*

