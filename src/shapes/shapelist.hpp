/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#include "shape.hpp"

class ShapeList : public IShape
{
public:
    ShapeList()
    {
    }

    ShapeList(IShape** l, int n)
        : list(l)
        , list_size(n)
    {
    }

    bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const override final
    {
        HitData temp_rec;
        bool hit_anything = false;

        float closest_so_far = t_max;
        int closest_index_so_far = -1;
        for (int i = 0; i < list_size; ++i)
        {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
                closest_index_so_far = i;
            }
        }

        if (hit_anything)
        {
            list[closest_index_so_far]->get_hit_data(r, rec);
        }

        return hit_anything;
    }

    void get_hit_data(const Ray& /* r */, HitData& /* rec */) const override final
    {
    }

private:
    IShape** list;
    int list_size;
};
