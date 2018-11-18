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
    ShapeList()
    {
    }
    
    ShapeList(IShape** l, int n, bool use_bbox = false)
        : list(l)
        , list_size(n)
        , bbox_hit(use_bbox)
    {
		for (int i = 0; i < list_size; ++i)
		{
            vec3 outmin, outmax;
            list[i]->get_bounds(outmin, outmax);

            bbox.expand(outmin);
            bbox.expand(outmax);
		}
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const override final
    {
        HitData temp_rec;
        bool hit_anything = false;

		// check for bbox hit before looping through all objects
		if (!bbox_hit || hit_bbox(r, t_min, t_max))
		{		
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
		}

        return hit_anything;
    }

    virtual void get_hit_data(const Ray& /* r */, HitData& /* rec */) const override final
    {
		// data already filled by hit()
    }

    virtual void get_bounds(vec3& min, vec3& max) const override final
    {
        bbox.get_bounds(min, max);
    }

private:
	bool hit_bbox(const Ray& r, float t_min, float t_max) const
	{
        HitData tmp;
		return bbox.hit(r, t_min, t_max, tmp);
	}

    IShape** list;
    int list_size;
    bool bbox_hit;
    Box bbox;
};
