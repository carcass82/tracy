/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "hitable.hpp"

class bvh_node : public hitable
{
public:
    bvh_node(hitable **l, int n, float time0, float time1)
    {
        int axis = int(3 * fastrand());

        auto x_comparer = [](const void* a, const void* b)
        {
            hitable* ah = *(hitable**)a;
            hitable* bh = *(hitable**)b;

            aabb box_left, box_right;
            if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right))
            {
                std::cerr << "no bbox in bvh_node constructor!\n";
            }

            return (box_left.vmin.x - box_right.vmin.x < 0.0f)? -1 : 1;
        };

        auto y_comparer = [](const void* a, const void* b)
        {
            hitable* ah = *(hitable**)a;
            hitable* bh = *(hitable**)b;

            aabb box_left, box_right;
            if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right))
            {
                std::cerr << "no bbox in bvh_node constructor!\n";
            }

            return (box_left.vmin.y - box_right.vmin.y < 0.0f)? -1 : 1;
        };

        auto z_comparer = [](const void* a, const void* b)
        {
            hitable* ah = *(hitable**)a;
            hitable* bh = *(hitable**)b;

            aabb box_left, box_right;
            if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right))
            {
                std::cerr << "no bbox in bvh_node constructor!\n";
            }

            return (box_left.vmin.z - box_right.vmin.z < 0.0f)? -1 : 1;
        };

        switch (axis)
        {
        case 0:
            qsort(l, n, sizeof(hitable*), x_comparer);
            break;

        case 1:
            qsort(l, n, sizeof(hitable*), y_comparer);
            break;

        case 2:
            qsort(l, n, sizeof(hitable*), z_comparer);
            break;
        }

        switch (n)
        {
        case 1:
            left = right = l[0];
            break;

        case 2:
            left = l[0];
            right = l[1];
            break;

        default:
            left = new bvh_node(l, n / 2, time0, time1);
            right = new bvh_node(l + n / 2, n - n / 2, time0, time1);
            break;
        }

        aabb box_left, box_right;
        if (!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right))
        {
            std::cerr << "no bbox in bvh_node constructor!\n";
        }

        box = box_left;
        box.expand(box_right);
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override
    {
        if (box.hit(r, t_min, t_max))
        {
            hit_record left_rec;
            bool hit_left = left->hit(r, t_min, t_max, left_rec);

            hit_record right_rec;
            bool hit_right = right->hit(r, t_min, t_max, right_rec);

            if (hit_left && hit_right)
            {
                if (left_rec.t < right_rec.t)
                {
                    rec = left_rec;
                }
                else
                {
                    rec = right_rec;
                }
                return true;
            }
            else if (hit_left)
            {
                rec = left_rec;
                return true;
            }
            else if (hit_right)
            {
                rec = right_rec;
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }

    virtual bool bounding_box(float t0, float t1, aabb& b) const override
    {
        b = box;
        return true;
    }

private:
    hitable* left;
    hitable* right;
    aabb box;
};
