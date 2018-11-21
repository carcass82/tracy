/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "shape.hpp"
#include "box.hpp"

class ShapeList : public IShape
{
public:
    static constexpr uint16_t MAX_DEPTH = 50;
    static constexpr uint32_t MAX_LEAF_OBJECTS = 50;

    ShapeList(IShape** objects, size_t num, uint16_t depth = 0)
        : current_depth(depth)
    {
        for (int i = 0; i < num; ++i)
        {
            vec3 outmin, outmax;
            objects[i]->get_bounds(outmin, outmax);

            bbox.expand(outmin);
            bbox.expand(outmax);
        }

        if (num < MAX_LEAF_OBJECTS)
        {
            list_size = num;
            list = new IShape*[list_size];
            memcpy(list, objects, list_size * sizeof(IShape*));
        }
        else
        {
            vec3 bmin, bmax;
            bbox.get_bounds(bmin, bmax);
            vec3 center = (bmin + bmax) / 2.f;
            
            Box bchildren[] = {
                { vec3(bmin.x, bmin.y, bmin.z),       vec3(center.x, center.y, center.z), nullptr },
                { vec3(center.x, bmin.y, bmin.z),     vec3(bmax.x, center.y, center.z),   nullptr },
                { vec3(bmin.x, center.y, bmin.z),     vec3(center.x, bmax.y, center.z),   nullptr },
                { vec3(center.x, center.y, center.z), vec3(bmax.x, bmax.y, bmin.z),       nullptr },
                { vec3(bmin.x, bmin.y, center.z),     vec3(center.x, center.y, bmax.z),   nullptr },
                { vec3(center.x, bmin.y, center.z),   vec3(bmax.x, center.y, bmax.z),     nullptr },
                { vec3(bmin.x, center.y, center.z),   vec3(center.x, bmax.y, bmax.z),     nullptr },
                { vec3(center.x, center.y, center.z), vec3(bmax.x, bmax.y, bmax.z),       nullptr }
            };

            for (size_t i = 0; i < array_size(bchildren); ++i)
            {
                std::vector<IShape*> objects_to_set;
                for (int j = 0; j < num; ++j)
                {
                    if (bchildren[i].contains(objects[j]))
                    {
                        objects_to_set.emplace_back(objects[j]);
                    }
                }

                children[i] = new ShapeList(&objects_to_set[0], objects_to_set.size(), depth + 1);
            }
        }
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const override final
    {
        if (hit_bbox(r, t_min, t_max))
        {
            if (has_children())
            {
                for (int i = 0; i < 8; ++i)
                {
                    if (children[i]->hit(r, t_min, t_max, rec))
                    {
                        return true;
                    }
                }
            }
            else
            {
                HitData temp_rec;
                bool hit_anything = false;

                {
                    float closest_so_far = t_max;
                    int closest_index_so_far = -1;
                    for (size_t i = 0; i < list_size; ++i)
                    {
                        if (list[i]->hit(r, t_min, closest_so_far, temp_rec))
                        {
                            hit_anything = true;
                            closest_so_far = temp_rec.t;
                            rec = temp_rec;
                            closest_index_so_far = int(i);
                        }
                    }

                    if (hit_anything)
                    {
                        list[closest_index_so_far]->get_hit_data(r, rec);
                    }
                }

                return hit_anything;
            }
        }

        return false;
    }

    virtual void get_hit_data(const Ray& /* r */, HitData& /* rec */) const override final
    {
    }

    virtual void get_bounds(vec3& min, vec3& max) const override final
    {
    }

    bool has_any_objects() const { return children[0] != nullptr || list_size > 0; }
    bool has_children() const { return children[0] != nullptr; }

private:
    bool hit_bbox(const Ray& r, float t_min, float t_max) const
    {
        HitData tmp;
        return bbox.hit(r, t_min, t_max, tmp);
    }

    ShapeList* children[8] = { nullptr };
    IShape** list = { nullptr };
    size_t list_size = 0;
    int current_depth = 0;
    Box bbox;
};
