/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "common.h"

struct Triangle
{
    CUDA_DEVICE_CALL Triangle()
    {}

    CUDA_DEVICE_CALL Triangle(const vec3& in_v0, const vec3& in_v1, const vec3& in_v2, uint16_t in_mesh, uint16_t in_triangle)
        : v{ in_v0, in_v1, in_v2 }
        , v0v1{ v[1] - v[0] }
        , v0v2{ v[2] - v[0] }
        , mesh_idx{ in_mesh }
        , tri_idx{ in_triangle }
    {}


    vec3 v[3]{};
    vec3 v0v1{};
    vec3 v0v2{};
    uint16_t mesh_idx{};
    uint16_t tri_idx{};
};
