/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once

#include "ext/cclib/cclib.h"
using cc::math::vec3;

#include "shapes/box.hpp"
#include "shapes/sphere.hpp"
#include "shapes/bvh_node.hpp"
#include "shapes/modifiers/translate.hpp"
#include "shapes/modifiers/rotate.hpp"
#include "materials/lambertian.hpp"
#include "materials/metal.hpp"
#include "materials/dielectric.hpp"
#include "materials/emissive.hpp"
#include "textures/image.hpp"
#include "textures/color.hpp"
#include "textures/checker.hpp"

hitable* random_scene()
{
    const int n = 500;
    hitable** list = new hitable*[n + 1];

    texture* terrain_texture = new checker_texture(new constant_texture(vec3(0.2f, 0.3f, 0.1f)), new constant_texture(vec3(0.9f, 0.9f, 0.9f)));
    list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(terrain_texture));

    int i = 1;
    for (int a = -10; a < 10; ++a) {
        for (int b = -10; b < 10; ++b) {
            float choose_mat = float(fastrand());
            vec3 center(a + 0.9f * fastrand(), 0.2f, b + 0.9f * fastrand());
            if (length(center - vec3(4.f, 0.2f, 0.f)) > 0.9f) {
                if (choose_mat < 0.8) {
                    //list[i++] = new moving_sphere(center, center + vec3(0, 0.5 * fastrand(), 0.0), 0.0, 1.0, 0.2, new lambertian(new constant_texture(vec3(fastrand() * fastrand(), fastrand() * fastrand(), fastrand() * fastrand())));
                    list[i++] = new sphere(center, 0.2f, new lambertian(new constant_texture(vec3(fastrand() * fastrand(), fastrand() * fastrand(), fastrand() * fastrand()))));
                } else if (choose_mat < 0.95) {
                    list[i++] = new sphere(center, 0.2f, new metal(vec3(0.5f * (1.0f + float(fastrand())), 0.5f * (1.0f + float(fastrand())), 0.5f * (1.0f + float(fastrand()))), 0.5f * float(fastrand())));
                } else {
                    list[i++] = new sphere(center, 0.2f, new dielectric(1.5));
                }

            }
        }
    }

    // area light
    //list[i++] = new xy_rect(-2, 6, 0, 3, -3, new emissive(new constant_texture(vec3(4,4,4))));
    list[i++] = new xz_rect(-20, 20, -20, 20, 10, new emissive(new constant_texture(vec3(2,2,2))));

    // lambertian
    list[i++] = new sphere(vec3(-2.f, 1.f, 0.f), 1.0f, new lambertian(new constant_texture(vec3(0.4f, 0.2f, 0.1f))));

    // dielectric
    list[i++] = new sphere(vec3(0.f, 1.f, 0.f), 1.0f, new dielectric(1.5f));

    // metal
    list[i++] = new sphere(vec3(2.f, 1.f, 0.f), 1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

    // lambertian noise ("marble like")
    //list[i++] = new sphere(vec3(4, 1, 0), 1.0, new lambertian(new noise_texture(5.0f)));

    // lambertian textured
    int nx, ny, nn;
    unsigned char* tex_data = stbi_load("data/earth.jpg", &nx, &ny, &nn, 0);
    list[i++] = new sphere(vec3(6, 1, 0), 1.0, new lambertian(new image_texture(tex_data, nx, ny)));

    return new hitable_list(list, i);
}

hitable* cornellbox_scene()
{
    const int n = 50;
    hitable** list = new hitable*[n + 1];

    material* red = new lambertian(new constant_texture(vec3(0.65f, 0.05f, 0.05f)));
    material* white = new lambertian(new constant_texture(vec3(0.73f, 0.73f, 0.73f)));
    material* green = new lambertian(new constant_texture(vec3(0.12f, 0.45f, 0.15f)));
    material* light = new emissive(new constant_texture(vec3(15.f, 15.f, 15.f)));

    int i = 0;
    list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
    list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
    list[i++] = new flip_normals(new xz_rect(213, 343, 227, 332, 554, light));
    list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
    list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
    list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));

    // large box
    list[i++] = new translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), white), 15), vec3(265, 0, 295));

    // small box
    list[i++] = new translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 165, 165), white), -18), vec3(130, 0, 65));

    material* alluminium = new metal(vec3(.8f, .85f, .88f), .05f);
    material* glass = new dielectric(1.5);
    material* gold = new metal(vec3(1.f, .71f, .29f), .05f);

    list[i++] = new sphere(vec3(130 + 82.5 - 25, 215, 65 + 82.5 - 25), 50.0, glass);
    list[i++] = new sphere(vec3(265 + 82.5 + 35, 400, 295 + 82.5 - 35), 70.0, alluminium);
    list[i++] = new sphere(vec3(265 + 82.5 + 15, 30, 80), 30.0, gold);

    return new hitable_list(list, i);
}

hitable* final()
{
    int nb = 20;
    hitable** list = new hitable*[30];
    hitable** boxlist = new hitable*[10000];
    hitable** boxlist2 = new hitable*[10000];

    material* white = new lambertian(new constant_texture(vec3(0.73f, 0.73f, 0.73f)));
    material* ground = new lambertian(new constant_texture(vec3(0.48f, 0.83f, 0.53f)));
    material* light = new emissive(new constant_texture(vec3(15.f, 15.f, 15.f)));

    int b = 0;
    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < nb; ++j) {
            float w = 100;
            float x0 = -1000 + i * w;
            float z0 = -1000 + j * w;
            float y0 = 0;
            float x1 = x0 + w;
            float y1 = 100 * (float(fastrand()) + 0.01f);
            float z1 = z0 + w;

            boxlist[b++] = new box(vec3(x0, y0, z0), vec3(x1, y1, z1), ground);
        }
    }

    //material* _lambertian = new lambertian(new constant_texture(vec3(0.7, 0.3, 0.1)));
    material* _dielectric = new dielectric(1.5f);
    material* _metal = new metal(vec3(0.8f, 0.8f, 0.9f), 10.0f);

    int nx, ny, nn;
    unsigned char* tex_data = stbi_load("data/earth.jpg", &nx, &ny, &nn, 0);
    material* _textured = new lambertian(new image_texture(tex_data, nx, ny));

    //material* _noise = new lambertian(new noise_texture(0.1f));


    int l = 0;
    list[l++] = new bvh_node(boxlist, b, 0, 1);
    list[l++] = new xz_rect(123, 423, 147, 412, 554, light);
    //list[l++] = new moving_sphere(vec3(400,400,200), vec3(430,400,200), 0, 1, 50, _lambertian);
    list[l++] = new sphere(vec3(260, 150, 45), 50, _dielectric);
    list[l++] = new sphere(vec3(0, 150, 145), 50, _metal);

    hitable* boundary = new sphere(vec3(360, 150, 145), 70, _dielectric);
    list[l++] = boundary;
    //list[l++] = new constant_medium(boundary, 0.2, new constant_texture(vec3(0.2, 0.4, 0.9)));

    boundary = new sphere(vec3(0, 0, 0), 5000, _dielectric);
    //list[l++] = new constant_medium(boundary, 0.0001, new constant_texture(vec3(1.0, 1.0, 1.0)));

    list[l++] = new sphere(vec3(400, 200, 400), 100, _textured);
    //list[l++] = new sphere(vec3(220, 280, 300), 80, _noise);

    int ns = 1000;
    for (int i = 0; i < ns; ++i) {
        boxlist2[i] = new sphere(vec3(165 * fastrand(), 165 * fastrand(), 165 * fastrand()), 10, white);
    }

    list[l++] = new translate(new rotate_y(new bvh_node(boxlist2, ns, 0.0, 1.0), 15), vec3(-100, 270, 395));

    return new hitable_list(list, l);
}

hitable* test_scene()
{
    hitable** list = new hitable*[30];


    //texture* ground = new checker_texture(new constant_texture(vec3(0.2, 0.2, 0.4)), new constant_texture(vec3(0.8, 0.8, 1.0)));
    //material* _lambertian = new lambertian(ground);
    //material* red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
    material* _light = new emissive(new constant_texture(vec3(50, 50, 50)));

    // lambertian textured
    int nx, ny, nn;
    unsigned char* tex_data = stbi_load("data/earth.jpg", &nx, &ny, &nn, 0);

    int i = 0;

    list[i++] = new sphere(vec3(1.f, 0.6f, 0.f), 0.3f, _light);
    //list[i++] = new sphere(vec3(0, -1, -1), 1, _lambertian);
    //list[i++] = new sphere(vec3(0, -1, -1), 1.0, new lambertian(new noise_texture(5.0f)));
    //list[i++] = new xz_rect(-50, 50, -50, 50, 0, red);
    //list[i++] = new yz_rect(0, 50, 0, 50, 0, red);
    list[i++] = new sphere(vec3(0.f, -1.f, -1.f), 1.0f, new lambertian(new image_texture(tex_data, nx, ny)));

    return new hitable_list(list, i);
}

hitable* first_scene()
{
    int n = 500;
    hitable** list = new hitable*[n + 1];

    int i = 0;
    list[i++] = new xz_rect(-15, 15, -15, 15, 0, new lambertian(new checker_texture(new constant_texture(vec3(0.7f, 0.7f, 0.7f)), new constant_texture(vec3(0.2f, 0.2f, 0.2f)))));

    int s = 0;
    hitable** spheres = new hitable*[1024];
    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            float choose_mat = fastrand();
            vec3 center(a + .9f * fastrand(), .2f, b + .9f * fastrand());
            if (length(center - vec3(4.f, .2f, .0f)) > .9f) {
                if (choose_mat < .8f) { // diffuse
                    spheres[s++] = new sphere(center, .2f, new lambertian(new constant_texture(vec3(fastrand() * fastrand(), fastrand() * fastrand(), fastrand() * fastrand()))));
                } else if (choose_mat < .9f) { // metal
                    spheres[s++] = new sphere(center, .2f, new metal(vec3(.5f * (1 + fastrand()), .5f * (1 + fastrand()), .5f * (1 + fastrand())), .5f * fastrand()));
                } else if (choose_mat < .95f) { // light
                    spheres[s++] = new sphere(center, .2f, new emissive(new constant_texture(vec3(5 * fastrand(), 5 * fastrand(), 5 * fastrand()))));
                } else {
                    spheres[s++] = new sphere(center, .2f, new dielectric(1.5f));
                }
            }
        }
    }
    list[i++] = new bvh_node(spheres, s, 0.0, 1.0);

    list[i++] = new sphere(vec3(.0f, 1.f, .0f), 1.f, new dielectric(1.5f));
    list[i++] = new sphere(vec3(-4.f, 1.f, .0f), 1.f, new lambertian(new constant_texture(vec3(.4f, .2f, .1f))));
    list[i++] = new sphere(vec3(4.f, 1.f, .0f), 1.f, new metal(vec3(.7f, .6f, .5f), .1f));

    // lights
    list[i++] = new xz_rect(-8, 8, -8, 8, 10, new emissive(new constant_texture(vec3(20, 20, 20))));
    list[i++] = new xy_rect(-15, 15, 8, 10, -14, new emissive(new constant_texture(vec3(10, 10, 10))));
    list[i++] = new rotate_y(new xy_rect(-15, 15, 8, 10, -14, new emissive(new constant_texture(vec3(10, 10, 10)))), 90);
    list[i++] = new flip_normals(new xy_rect(-15, 15, 8, 10, 14, new emissive(new constant_texture(vec3(10, 10, 10)))));
    list[i++] = new flip_normals(new rotate_y(new xy_rect(-15, 15, 8, 10, 19, new emissive(new constant_texture(vec3(10, 10, 10)))), 90));

    // sides
    list[i++] = new xy_rect(-15, 15, -15, 15, -15, new lambertian(new constant_texture(vec3(0.9f, 0.0f, 0.0f))));
    list[i++] = new rotate_y(new xy_rect(-15, 15, -15, 15, -15, new lambertian(new constant_texture(vec3(0.0f, 0.9f, 0.0f)))), 90);
    list[i++] = new flip_normals(new xy_rect(-15, 15, -15, 15, 15, new lambertian(new constant_texture(vec3(0.0f, 0.0f, 0.9f)))));
    list[i++] = new flip_normals(new rotate_y(new xy_rect(-15, 15, -15, 15, 20, new lambertian(new constant_texture(vec3(0.0f, 0.9f, 0.9f)))), 90));

    // top
    list[i++] = new xz_rect(-15, 15, -15, 15, 12, new lambertian(new constant_texture(vec3(0.9f, 0.0f, 0.9f))));

    return new hitable_list(list, i);
}

hitable* sort_by_distance(hitable* list, const vec3& point)
{
    hitable_list* scene = (hitable_list*)list;

    for (int i = 0; i < scene->list_size; ++i) {

        for (int j = scene->list_size - 1; j > i; --j) {

            aabb bbox1, bbox2;
            scene->list[j - 1]->bounding_box(0.001f, FLT_MAX, bbox1);
            scene->list[j]->bounding_box(0.001f, FLT_MAX, bbox2);
            float dist1 = bbox1.distance(point);
            float dist2 = bbox2.distance(point);

            if (dist2 < dist1)
            {
                std::swap(scene->list[j], scene->list[j - 1]);
            }
        }
    }

    return list;
}

enum eScene { eRANDOM, eCORNELLBOX, eFINAL, eTEST, eFROMFILE, eFIRST_SCENE, eNUM_SCENES };

hitable* load_scene(eScene scene, camera& cam, float ratio)
{
    switch (scene) {
    case eRANDOM:
        std::cerr << "'random' scene selected\n";
        cam.setup(vec3(9.0f, 1.5f, 6.0f), vec3(2.0f, 0.5f, -2.0f), vec3(0.0f, 1.0f, 0.0f), 45.0f, ratio, 2.0f, 5.0f);
        return random_scene();

    case eCORNELLBOX:
        std::cerr << "'cornell' scene selected\n";
        cam.setup(vec3(278, 278, -800), vec3(278, 278, 0), vec3(0.0f, 1.0f, 0.0f), 40.0f, ratio, 0.0f, 10.0f);
        return cornellbox_scene();

    case eFINAL:
        std::cerr << "'final' scene selected\n";
        cam.setup(vec3(278, 278, -800), vec3(278, 278, 0), vec3(0.0f, 1.0f, 0.0f), 40.0f, ratio, 0.0f, 10.0f);
        return final();

    case eTEST:
        std::cerr << "'test' scene selected\n";
        cam.setup(vec3(0, 0, 5), vec3(0, 0, 0), vec3(0, 1, 0), 45.0f, ratio, 0.0f, 10.0f);
        return test_scene();

    case eFIRST_SCENE:
        std::cerr << "'first' scene selected\n";
        cam.setup(vec3(17, 2, 7), vec3(0, 0, -1), vec3(0, 1, 0), 20.0f, ratio, 2.0f, 20.0f);
        return first_scene();

    default:
        std::cerr << "tracing NULL, i'm going to crash...\n";
        return nullptr;
    };
}
