/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include "vertex.h"
using Vertex = BaseVertex<true>; /* with tangent and bitangent */
using Index = unsigned int;