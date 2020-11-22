/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include "common.h"
#include "ray.h"

namespace collision { struct HitData; }

class Material
{
public:
	enum MaterialID { eINVALID, eLAMBERTIAN, eMETAL, eDIELECTRIC, eEMISSIVE };

    CUDA_DEVICE_CALL Material()
    {}

    CUDA_DEVICE_CALL Material(MaterialID in_type, const vec3& in_albedo, float in_roughness = .0f, float in_ior = 1.f)
        : material_type_(in_type)
        , albedo_(in_albedo)
        , roughness_(in_roughness)
        , ior_(in_ior)
    {}

    CUDA_DEVICE_CALL bool Scatter(const Ray& ray, const collision::HitData& hit, vec3& out_attenuation, vec3& out_emission, Ray& out_scattered, RandomCtx random_ctx) const;

    MaterialID GetType() const { return material_type_; }

    const vec3& GetAlbedo() const { return albedo_; }

    float GetRoughness() const { return roughness_; }

    float GetIOR() const { return ior_; }

private:
    MaterialID material_type_{ eINVALID };
    vec3 albedo_{};
    float roughness_{ .0f };
    float ior_{ 1.f };
};
