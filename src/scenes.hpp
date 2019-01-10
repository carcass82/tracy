/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "ext/tiny_obj_loader.h"

#include "shapes/shapelist.hpp"
#include "shapes/box.hpp"
#include "shapes/sphere.hpp"
#include "shapes/triangle.hpp"

#include "materials/lambertian.hpp"
#include "materials/metal.hpp"
#include "materials/dielectric.hpp"
#include "materials/emissive.hpp"

#include "textures/bitmap.hpp"
#include "textures/constant.hpp"
#include "textures/checker.hpp"


IShape* load_mesh(const char* obj_path)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, obj_path))
    {
        std::cerr << "unable to load model '" << obj_path << "'\n";
        return nullptr;
    }

    IMaterial* green = new Lambertian(new Constant(vec3(.05f, .85f, .02f)));
    
    IShape** tmplist = new IShape*[attrib.vertices.size()];
    
    constexpr unsigned MAX_SPLIT = 50;
    IShape** list = new IShape*[MAX_SPLIT];
    
    vec3 verts[3];
    vec3 norms[3];
    vec2 uvs[3];
    unsigned int i = 0;
    unsigned int n = 0;
    unsigned int v = 0;
    for (const tinyobj::shape_t& shape : shapes)
    {
        for (const tinyobj::index_t& index : shape.mesh.indices)
        {
            float v0 = attrib.vertices[3 * index.vertex_index + 0];
            float v1 = attrib.vertices[3 * index.vertex_index + 1];
            float v2 = attrib.vertices[3 * index.vertex_index + 2];

            float n0 = attrib.vertices[3 * index.normal_index + 0];
            float n1 = attrib.vertices[3 * index.normal_index + 1];
            float n2 = attrib.vertices[3 * index.normal_index + 2];

            float s = attrib.vertices[2 * index.texcoord_index + 0];
            float t = attrib.vertices[2 * index.texcoord_index + 1];

            norms[v] = vec3{ n0, n1, n2 };
            uvs[v] = vec2{ s, t };
            verts[v++] = vec3{ v0, v1, v2 };

            if (v == 3)
            {
                list[i++] = new Triangle(verts[0], verts[1], verts[2],
                                         norms[0], norms[1], norms[2],
                                         uvs[0], uvs[1], uvs[2],
                                         green);
                v = 0;
            }

            if (i >= MAX_SPLIT)
            {
                tmplist[n++] = new ShapeList(list, i);
                
                i = 0;
                list = new IShape*[MAX_SPLIT];
            }
        }
    }

    if (i > 0)
    {
        tmplist[n++] = new ShapeList(list, i);
    }

    return new ShapeList(tmplist, n);
}

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
    list[i++] = new Box(vec3(-20.f, 10.f, -20.f), vec3(20.f, 10.1f, 20.f), new Emissive(new Constant(vec3(2, 2, 2))));

    // lambertian
    list[i++] = new Sphere(vec3(0.f, 1.f, 0.f), 1.0f, new Lambertian(new Constant(vec3(0.4f, 0.2f, 0.1f))));

    // dielectric
    list[i++] = new Sphere(vec3(2.f, 1.f, 0.f), 1.0f, new Dielectric(1.5f));

    // metal
    list[i++] = new Sphere(vec3(4.f, 1.f, 0.f), 1.0f, new Metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));
    
    // lambertian textured
    int nx, ny, nn;
    unsigned char* tex_data = stbi_load("../data/earth.jpg", &nx, &ny, &nn, 0);
    list[i++] = new Sphere(vec3(6.f, 1.f, 0.f), 1.0, new Lambertian(new Bitmap(tex_data, nx, ny)));

    return new ShapeList(list, i);
}

IShape* cornellbox_scene()
{
    const int n = 500;
    IShape** list = new IShape*[n + 1];

    IMaterial* red = new Lambertian(new Constant(vec3(0.65f, 0.05f, 0.05f)));
    IMaterial* white = new Lambertian(new Constant(vec3(0.73f, 0.73f, 0.73f)));
    IMaterial* green = new Lambertian(new Constant(vec3(0.12f, 0.45f, 0.15f)));
    IMaterial* light = new Emissive(new Constant(vec3(15.f, 15.f, 15.f)));
    
    int i = 0;
    list[i++] = new Box(vec3(213.f, 554.f, 227.f), vec3(343.f, 555.f, 332.f), light);
    list[i++] = new Box(vec3(555.f, .0f, 0.f), vec3(555.1f, 555.f, 555.f), green);
    list[i++] = new Box(vec3(-0.1f, .0f, 0.f), vec3(.0f, 555.f, 555.f), red);
    list[i++] = new Box(vec3(.0f, -.1f, 0.f), vec3(555.f, 0.f, 555.f), white);      // floor
    list[i++] = new Box(vec3(.0f, 555.f, 0.f), vec3(555.f, 555.1f, 555.f), white);  // roof
    list[i++] = new Box(vec3(.0f, .0f, 554.9f), vec3(555.f, 555.f, 555.f), white);  // back
    list[i++] = new Box(vec3(265.f, .0f, 295.f), vec3(430.f, 330.f, 460.f), white); // large box
    list[i++] = new Box(vec3(130.f, .0f, 65.f), vec3(295.f, 165.f, 230.f), white);  // small box

#if 0 // old version, with rects / flip_normals / rotations / translations
    //list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
    //list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
    //list[i++] = new flip_normals(new xz_rect(213, 343, 227, 332, 554, light));
    //list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
    //list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
    //list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));
    
    // large box
    //list[i++] = new translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), white), 15), vec3(265, 0, 295));
    
    // small box
    //list[i++] = new translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 165, 165), white), -18), vec3(130, 0, 65));
#endif

    // "custom" spheres to test materials
    IMaterial* alluminium = new Metal(vec3(.8f, .85f, .88f), .05f);
    IMaterial* glass = new Dielectric(1.5);
    IMaterial* gold = new Metal(vec3(1.f, .71f, .29f), .05f);

    list[i++] = new Sphere(vec3(130 + 82.5 - 25, 215, 65 + 82.5 - 25), 50.0, glass);
    list[i++] = new Sphere(vec3(265 + 82.5 + 35, 400, 295 + 82.5 - 35), 70.0, alluminium);
    list[i++] = new Sphere(vec3(265 + 82.5 + 15, 30, 80), 30.0, gold);

    return new ShapeList(list, i);
}

IShape* gpu_scene()
{
    IMaterial* light = new Emissive(new Constant(vec3(5.f, 5.f, 5.f)));
    IMaterial* blue = new Lambertian(new Constant(vec3(0.1f, 0.2f, 0.5f)));
    IMaterial* red = new Lambertian(new Constant(vec3(.85f, .05f, .02f)));
	IMaterial* green = new Lambertian(new Constant(vec3(.05f, .85f, .02f)));
    IMaterial* grey = new Lambertian(new Constant(vec3(0.2f, 0.2f, 0.2f)));
    IMaterial* glass = new Dielectric(1.5);
    IMaterial* alluminium = new Metal(vec3(.91f, .92f, .92f), .0f);
    IMaterial* gold = new Metal(vec3(1.f, .71f, .29f), .05f);
    IMaterial* copper = new Metal(vec3(.95f, .64f, .54f), .2f);

    int i = 0;
    IShape** objects = new IShape*[50];
    objects[i++] = new Sphere(vec3(0.f, 0.f, -1.f), .5f, blue);
    objects[i++] = new Sphere(vec3(0.f, 150.f, -1.f), 100.f, light);
    objects[i++] = new Sphere(vec3(1.f, 0.f, -1.f), .5f, alluminium);
    objects[i++] = new Sphere(vec3(-1.f, 0.f, -1.f), .5f, glass);
    objects[i++] = new Sphere(vec3(0.f, 0.f, 0.f), .2f, copper);
    objects[i++] = new Sphere(vec3(0.f, 1.f, -1.5f), .3f, gold);
    objects[i++] = new Sphere(vec3(0.f, 0.f, -2.5f), .5f, red);
    
    objects[i++] = new Box(vec3(-4.f, -0.5f, -3.1f), vec3(4.f, 2.f, -3.f), grey);
    objects[i++] = new Box(vec3(-4.f, -0.5f, 1.6f), vec3(4.f, 2.f, 1.7f), grey);
    objects[i++] = new Box(vec3(-4.f, -0.6f, -3.f), vec3(4.f, -0.5f, 1.7f), grey);
    objects[i++] = new Box(vec3(-4.1f, -0.5f, -3.f), vec3(-4.f, 2.f, 1.7f), grey);
    objects[i++] = new Box(vec3(4.f, -0.5f, -3.f), vec3(4.1f, 2.f, 1.7f), grey);
    
    objects[i++] = new Box(vec3(-1.8f, 1.f, -3.f), vec3(1.8f, 1.1f, -2.9f), light);
    objects[i++] = new Box(vec3(-1.8f, 1.f, 1.6f), vec3(1.8f, 1.1f, 1.61f), light);
    
    objects[i++] = new Triangle(vec3(-1.f, .5f, -2.5f), vec3(1.f, .5f, -2.5f), vec3(1.f, 1.5f, -2.5f), green);

    //objects[i++] = load_mesh("./data/teapot.obj");

    return new ShapeList(objects, i);
}

enum eScene { eRANDOM, eCORNELLBOX, eTESTGPU, eNUM_SCENES };

IShape* load_scene(eScene scene, Camera& cam, float ratio)
{
    switch (scene) {
    case eRANDOM:
        std::cerr << "'random' scene selected\n";
        cam.setup(vec3(9.0f, 1.5f, 6.0f), vec3(2.0f, 0.5f, -2.0f), vec3(0.0f, 1.0f, 0.0f), 45.0f, ratio);
        return random_scene();

    case eCORNELLBOX:
        std::cerr << "'cornell' scene selected\n";
        cam.setup(vec3(278, 278, -800), vec3(278, 278, 0), vec3(0.0f, 1.0f, 0.0f), 40.0f, ratio);
        return cornellbox_scene();

    case eTESTGPU:
        std::cerr << "'testGPU' scene selected\n";
        cam.setup(vec3(-.5f, 1.2f, 1.5f), vec3(.0f, .0f, -1.f), vec3(0.0f, 1.0f, 0.0f), 60.0f, ratio);
        return gpu_scene();

    default:
        std::cerr << "tracing NULL, i'm going to crash...\n";
        return nullptr;
    };
}
