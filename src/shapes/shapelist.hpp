/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "shape.hpp"
#include "sphere.hpp"
#include "box.hpp"
#include "triangle.hpp"

class ShapeList : public IShape
{
public:
    ShapeList()
    {
    }

    ShapeList(IShape** objects, int num)
    {
        boxes = new Box[num];
        spheres = new Sphere[num];
        triangles = new Triangle[num];
        lists = new ShapeList[num];

        for (int i = 0; i < num; ++i)
        {
            vec3 outmin, outmax;
            objects[i]->get_bounds(outmin, outmax);
            bbox.expand(outmin);
            bbox.expand(outmax);

            switch (objects[i]->get_id())
            {
            case make_id('B','O','X'):
                boxes[boxes_count++] = *static_cast<Box*>(objects[i]);
                break;
            case make_id('S', 'P', 'H'):
                spheres[spheres_count++] = *static_cast<Sphere*>(objects[i]);
                break;
            case make_id('T', 'R', 'I'):
                triangles[triangles_count++] = *static_cast<Triangle*>(objects[i]);
                break;
            case make_id('L', 'I', 'S', 'T'):
                lists[list_count++] = *static_cast<ShapeList*>(objects[i]);
            default:
                break;
            }
        }

        //
        // cleanup memory of original objects
        //
        for (int i = 0; i < num; ++i)
        {
            delete objects[i];
        }
        delete objects;
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const override final
    {
        if (hit_bbox(r, t_min, t_max))
        {
            HitData temp_rec;
            temp_rec.t = t_max;
            int closest_index_so_far = -1;

            //
            // check other lists
            //
            bool hit_list = false;
            for (int i = 0; i < list_count; ++i)
            {
                hit_list = hit_list | lists[i].hit(r, t_min, temp_rec.t, temp_rec);
            }
            //
            // check other lists
            //

            //
            // check spheres
            //
            bool hit_sphere = false;
            for (int i = 0; i < spheres_count; ++i)
            {
                vec3 oc{ r.get_origin() - spheres[i].center };
                float b = dot(oc, r.get_direction());
                float c = dot(oc, oc) - spheres[i].radius2;

                if (b <= .0f || c <= .0f)
                {
                    float discriminant = b * b - c;
                    if (discriminant > .0f)
                    {
                        discriminant = sqrtf(discriminant);

                        float t0 = -b - discriminant;
                        if (t0 < temp_rec.t && t0 > t_min)
                        {
                            temp_rec.t = t0;
                            closest_index_so_far = i;
                            hit_sphere = true;
                        }

                        float t1 = -b + discriminant;
                        if (t1 < temp_rec.t && t1 > t_min)
                        {
                            temp_rec.t = t1;
                            closest_index_so_far = i;
                            hit_sphere = true;
                        }
                    }
                }
            }
            if (hit_sphere)
            {
                spheres[closest_index_so_far].get_hit_data(r, temp_rec);
            }
            //
            // check spheres
            //

            //
            // check boxes
            //
            bool hit_box = false;
            const vec3 inv_ray = 1.f / r.get_direction();
            for (int i = 0; i < boxes_count; ++i)
            {
                float tmin = t_min;
                float tmax = temp_rec.t;
                bool hit = false;

                const vec3 minbound = (boxes[i].pmin - r.get_origin()) * inv_ray;
                const vec3 maxbound = (boxes[i].pmax - r.get_origin()) * inv_ray;
                for (int dim = 0; dim < 3; ++dim)
                {
                    float t1 = minbound[dim];
                    float t2 = maxbound[dim];

                    tmin = max(tmin, min(t1, t2));
                    tmax = min(tmax, max(t1, t2));

                    if (tmin > tmax || tmin > temp_rec.t)
                    {
                        hit = false;
                        break;
                    }
                    hit = true;
                }

                if (hit)
                {
                    temp_rec.t = tmin;
                    closest_index_so_far = i;
                    hit_box = true;
                }
            }
            if (hit_box)
            {
                boxes[closest_index_so_far].get_hit_data(r, temp_rec);
            }
            //
            // check boxes
            //

            //
            // check triangles
            //
            bool hit_triangle = false;
            for (int i = 0; i < triangles_count; ++i)
            {
                vec3 pvec = cross(r.get_direction(), triangles[i].v0v2);
                float det = dot(triangles[i].v0v1, pvec);

                // if the determinant is negative the triangle is backfacing
                // if the determinant is close to 0, the ray misses the triangle
                if (det < 1e-6)
                {
                    continue;
                }

                float invDet = 1.f / det;

                vec3 tvec = r.get_origin() - triangles[i].vertices[0];
                float u = dot(tvec, pvec) * invDet;
                if (u < .0f || u > 1.f)
                {
                    continue;
                }

                vec3 qvec = cross(tvec, triangles[i].v0v1);
                float v = dot(r.get_direction(), qvec) * invDet;
                if (v < .0f || u + v > 1.f)
                {
                    continue;
                }

                float t = dot(triangles[i].v0v2, qvec) * invDet;
                if (t < temp_rec.t && t > t_min)
                {
                    temp_rec.t = dot(triangles[i].v0v2, qvec) * invDet;
                    temp_rec.uv = vec2{ u, v };
                    closest_index_so_far = i;
                    hit_triangle = true;
                }
            }
            if (hit_triangle)
            {
                triangles[closest_index_so_far].get_hit_data(r, temp_rec);
            }
            //
            // check triangles
            //

            if (hit_list || hit_sphere || hit_box || hit_triangle)
            {
                rec = temp_rec;
                return true;
            }
        }

        return false;
    }

    virtual void get_hit_data(const Ray& /* r */, HitData& /* rec */) const override final
    {
    }

    virtual void get_bounds(vec3& min, vec3& max) const override final
    {
        return bbox.get_bounds(min, max);
    }

    virtual uint32_t get_id() const override final
    {
        return make_id('L','I','S','T');
    }

private:
    bool hit_bbox(const Ray& r, float t_min, float t_max) const
    {
        const vec3 inv_ray = 1.f / r.get_direction();
        const vec3 minbound = (bbox.pmin - r.get_origin()) * inv_ray;
        const vec3 maxbound = (bbox.pmax - r.get_origin()) * inv_ray;

        float tmin = t_min;
        float tmax = t_max;
        for (int dim = 0; dim < 3; ++dim)
        {
            float t1 = minbound[dim];
            float t2 = maxbound[dim];

            tmin = max(tmin, min(t1, t2));
            tmax = min(tmax, max(t1, t2));

            if (tmin > tmax || tmin > t_max)
            {
                return false;
            }
        }

        return true;
    }

    Box* boxes = nullptr;
    Sphere* spheres = nullptr;
    Triangle* triangles = nullptr;
    ShapeList* lists = nullptr;
    int boxes_count = 0;
    int spheres_count = 0;
    int triangles_count = 0;
    int list_count = 0;

    Box bbox;
};
