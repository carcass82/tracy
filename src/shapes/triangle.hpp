/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "shape.hpp"

class Triangle : public IShape
{
public:
    Triangle()
    {
    }

    Triangle(vec3 v1, vec3 v2, vec3 v3, IMaterial* ptr)
        : vertices{ v1, v2, v3 }
        , mat(ptr)
    {
        v0v1 = vertices[1] - vertices[0];
        v0v2 = vertices[2] - vertices[0];
        normal[0] = normal[1] = normal[2] = normalize(cross(v0v1, v0v2));
        uv[0] = vec2{ .0f, .0f }; 
        uv[1] = vec2{ 1.f, .0f };
        uv[2] = vec2{ .0f, 1.f };
    }

    Triangle(vec3 v1, vec3 v2, vec3 v3,
             vec3 n1, vec3 n2, vec3 n3,
             IMaterial* ptr)
        : vertices{ v1, v2, v3 }
        , normal{ n1, n2, n3 }
        , mat(ptr)
    {
        v0v1 = vertices[1] - vertices[0];
        v0v2 = vertices[2] - vertices[0];
        uv[0] = vec2{ .0f, .0f };
        uv[1] = vec2{ 1.f, .0f };
        uv[2] = vec2{ .0f, 1.f };
    }

    Triangle(vec3 v1, vec3 v2, vec3 v3,
             vec3 n1, vec3 n2, vec3 n3,
             vec2 uv1, vec2 uv2, vec2 uv3,
             IMaterial* ptr)
        : vertices{ v1, v2, v3 }
        , normal{ n1, n2, n3 }
        , uv{ uv1, uv2, uv3 }
        , mat(ptr)
    {
        v0v1 = vertices[1] - vertices[0];
        v0v2 = vertices[2] - vertices[0];
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const override final
    {
        // see ShapeList
        return false;
    }

    virtual void get_hit_data(const Ray& r, HitData& rec) const override final
    {
        rec.point = r.point_at(rec.t);
        rec.normal = (1.f - rec.uv.x - rec.uv.y) * normal[0] + rec.uv.x * normal[1] + rec.uv.y * normal[2];
        rec.uv = (1.f - rec.uv.x - rec.uv.y) * uv[0] + rec.uv.x * uv[1] + rec.uv.y * uv[2];
        rec.material = mat;
    }

    virtual void get_bounds(vec3& min, vec3& max) const override final
    {
        min = min3(vertices[0], min3(vertices[1], vertices[2]));
        max = max3(vertices[0], max3(vertices[1], vertices[2]));
    }

    virtual uint32_t get_id() const override final
    {
        return make_id('T', 'R', 'I');
    }

    friend class ShapeList;

private:
    vec3 vertices[3];
    vec3 normal[3];
    vec2 uv[3];
    vec3 v0v1;
    vec3 v0v2;
    IMaterial* mat;
};
