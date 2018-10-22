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

#include "materials/lambertian.hpp"
#include "materials/metal.hpp"
#include "materials/dielectric.hpp"
#include "materials/emissive.hpp"

#include "textures/bitmap.hpp"
#include "textures/constant.hpp"
#include "textures/checker.hpp"


IShape* random_scene()
{
    const int n = 500;
    int i = 0;
    IShape** list = new IShape*[n + 1];

    ITexture* terrain_texture = new Checker(new Constant(vec3(0.2f, 0.3f, 0.1f)), new Constant(vec3(0.9f, 0.9f, 0.9f)));
    list[i++] = new Sphere(vec3(0, -1000, 0), 1000, new Lambertian(terrain_texture));

    for (int a = -10; a < 10; ++a)
    {
        for (int b = -10; b < 10; ++b)
        {
            float choose_mat = fastrand();

            vec3 center(a + 0.9f * fastrand(), 0.2f, b + 0.9f * fastrand());
            if (length(center - vec3(4.f, 0.2f, 0.f)) > 0.9f)
            {
                if (choose_mat < 0.8)
                {
                    list[i++] = new Sphere(center, 0.2f, new Lambertian(new Constant(vec3(fastrand() * fastrand(), fastrand() * fastrand(), fastrand() * fastrand()))));
                }
                else if (choose_mat < 0.95)
                {
                    list[i++] = new Sphere(center, 0.2f, new Metal(vec3(0.5f * (1.0f + float(fastrand())), 0.5f * (1.0f + float(fastrand())), 0.5f * (1.0f + float(fastrand()))), 0.5f * float(fastrand())));
                }
                else
                {
                    list[i++] = new Sphere(center, 0.2f, new Dielectric(1.5));
                }

            }
        }
    }

    
    // area light
    //list[i++] = new flip_normals(new xz_rect(-20, 20, -20, 20, 10, new emissive(new Constant(vec3(2,2,2)))));

    // lambertian
    list[i++] = new Sphere(vec3(0.f, 1.f, 0.f), 1.0f, new Lambertian(new Constant(vec3(0.4f, 0.2f, 0.1f))));

    // dielectric
    list[i++] = new Sphere(vec3(2.f, 1.f, 0.f), 1.0f, new Dielectric(1.5f));

    // metal
    list[i++] = new Sphere(vec3(4.f, 1.f, 0.f), 1.0f, new Metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));
    
    // lambertian textured
    int nx, ny, nn;
    unsigned char* tex_data = stbi_load("data/earth.jpg", &nx, &ny, &nn, 0);
    list[i++] = new Sphere(vec3(6.f, 1.f, 0.f), 1.0, new Lambertian(new Bitmap(tex_data, nx, ny)));

    return new ShapeList(list, i);
}

IShape* cornellbox_scene()
{
    /*
    const int n = 500;
    hitable** list = new hitable*[n + 1];

    material* red = new Lambertian(new constant_texture(vec3(0.65f, 0.05f, 0.05f)));
    material* white = new Lambertian(new constant_texture(vec3(0.73f, 0.73f, 0.73f)));
    material* green = new Lambertian(new constant_texture(vec3(0.12f, 0.45f, 0.15f)));
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

    // "custom" spheres to test materials
    material* alluminium = new Metal(vec3(.8f, .85f, .88f), .05f);
    material* glass = new Dielectric(1.5);
    material* gold = new Metal(vec3(1.f, .71f, .29f), .05f);

    list[i++] = new Sphere(vec3(130 + 82.5 - 25, 215, 65 + 82.5 - 25), 50.0, glass);
    list[i++] = new Sphere(vec3(265 + 82.5 + 35, 400, 295 + 82.5 - 35), 70.0, alluminium);
    list[i++] = new Sphere(vec3(265 + 82.5 + 15, 30, 80), 30.0, gold);

    return new ShapeList(list, i);
    */

    return nullptr;
}

IShape* gpu_scene()
{
    const int n = 50;
    int i = 0;
    IShape** list = new IShape*[n];

    IMaterial* light = new Emissive(new Constant(vec3(2.f, 2.f, 2.f)));
    IMaterial* blu = new Lambertian(new Constant(vec3(0.1f, 0.2f, 0.5f)));
    IMaterial* red = new Lambertian(new Constant(vec3(.85f, .05f, .02f)));
    IMaterial* green = new Lambertian(new Constant(vec3(.05f, .85f, .02f)));
    IMaterial* grey = new Lambertian(new Constant(vec3(0.2f, 0.2f, 0.2f)));
    IMaterial* glass = new Dielectric(1.5);
    IMaterial* alluminium = new Metal(vec3(.8f, .85f, .88f), .0f);
    IMaterial* gold = new Metal(vec3(1.f, .71f, .29f), .05f);

    list[i++] = new Sphere(vec3(0.f, 0.f, -1.f), .5f, blu);
    list[i++] = new Sphere(vec3(0.f, -100.5f, -1.f), 100.f, grey);
    list[i++] = new Sphere(vec3(1.f, 0.f, -1.f), .5f, alluminium);
    list[i++] = new Sphere(vec3(-1.f, 0.f, -1.f), .5f, glass);
    list[i++] = new Sphere(vec3(0.f, 150.f, -1.f), 100.f, light);
    list[i++] = new Sphere(vec3(0.f, 0.f, 0.f), .2f, glass);
    list[i++] = new Sphere(vec3(0.f, 1.f, -1.5f), .3f, gold);
    list[i++] = new Sphere(vec3(0.f, 0.f, -2.5f), .5f, red);

    list[i++] = new Box(vec3(-2.f, 0.f, -3.1f), vec3(2.f, 2.f, -3.f), green);

    return new ShapeList(list, i);
}

enum eScene { eRANDOM, eCORNELLBOX, eTESTGPU, eNUM_SCENES };

IShape* load_scene(eScene scene, Camera& cam, float ratio)
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

    case eTESTGPU:
        std::cerr << "'testGPU' scene selected\n";
        cam.setup(vec3(-.5f, 1.2f, 1.5f), vec3(.0f, .0f, -1.f), vec3(0.0f, 1.0f, 0.0f), 60.0f, ratio, 0.0f, 10.0f);
        return gpu_scene();

    default:
        std::cerr << "tracing NULL, i'm going to crash...\n";
        return nullptr;
    };
}
