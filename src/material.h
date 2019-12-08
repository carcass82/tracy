/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include "common.h"
#include "ray.h"

class Material
{
public:
	enum class MaterialID { eINVALID, eLAMBERTIAN, eMETAL, eDIELECTRIC, eEMISSIVE };

    Material()
        : material_type_(MaterialID::eINVALID)
        , albedo_(vec3{})
        , roughness_(.0f)
        , ior_(1.f)
    {}

    Material(MaterialID in_type, const vec3& in_albedo, float in_roughness = .0f, float in_ior = 1.f)
        : material_type_(in_type)
        , albedo_(in_albedo)
        , roughness_(in_roughness)
        , ior_(in_ior)
    {}

    bool Scatter(const Ray& ray, const HitData& hit, vec3& out_attenuation, vec3& out_emission, Ray& out_scattered) const;

    MaterialID GetType() const { return material_type_; }

    const vec3& GetAlbedo() const { return albedo_; }

    float GetRoughness() const { return roughness_; }

    float GetIOR() const { return ior_; }

private:
    MaterialID material_type_;
    vec3 albedo_;
    float roughness_;
    float ior_;
};
