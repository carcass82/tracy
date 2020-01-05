/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

struct GLMaterial
{
    GLMaterial(const Material& material)
        : albedo{ material.GetAlbedo() }
        , metalness{ material.GetType() == Material::eMETAL? 1.f : .0f }
        , roughness{ material.GetRoughness() }
        , ior{ material.GetIOR() }
    {}

    vec3 albedo;
    float metalness;
    float roughness;
    float ior;
};
