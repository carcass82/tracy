/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include <unordered_map>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include "ext/tiny_obj_loader.h"

#if !defined(USE_CUDA)
#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb_image.h"

#include "camera.hpp"

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


struct Scene
{
    Camera cam;
    IShape* world;
};
#else
struct DScene
{
    DCamera cam;

    DTriangle** h_triangles;
    int num_triangles;

    DBox** h_boxes;
    int num_boxes;

    DSphere** h_spheres;
    int num_spheres;
};
#endif

#if !defined(USE_CUDA)
IShape* load_mesh(const char* obj_path, IMaterial* obj_material)
{
#else
std::vector<DTriangle*> load_mesh(const char* obj_path, DMaterial* obj_material)
{
    std::vector<DTriangle*> mesh;
#endif

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, obj_path))
    {
        fprintf(stderr, "unable to load model '%s'\n", obj_path);
#if !defined(USE_CUDA)
        return nullptr;
#else
        return mesh;
#endif
    }

#if !defined(USE_CUDA)
    IShape** tmplist = new IShape*[attrib.vertices.size()];
    
    constexpr unsigned MAX_SPLIT = 100;
    IShape** list = new IShape*[MAX_SPLIT];
    
    vec3 verts[3];

    unsigned int i = 0;
    unsigned int n = 0;
#else
    float3 verts[3];
#endif
    unsigned int v = 0;
    for (const tinyobj::shape_t& shape : shapes)
    {
        for (const tinyobj::index_t& index : shape.mesh.indices)
        {
            float v0 = attrib.vertices[3 * index.vertex_index + 0];
            float v1 = attrib.vertices[3 * index.vertex_index + 1];
            float v2 = attrib.vertices[3 * index.vertex_index + 2];

#if !defined(USE_CUDA)
            verts[v++] = vec3{ v0, v1, v2 };

            if (v == 3)
            {
                list[i++] = new Triangle(verts[0], verts[1], verts[2], obj_material);
                v = 0;
            }

            if (i >= MAX_SPLIT)
            {
                tmplist[n++] = new ShapeList(list, i);
                
                i = 0;
                list = new IShape*[MAX_SPLIT];
            }
#else
            verts[v++] = make_float3(v0, v1, v2);

            if (v == 3)
            {
                mesh.push_back(triangle_create(verts[0], verts[1], verts[2], *obj_material));
                v = 0;
            }
#endif
        }
    }

#if !defined(USE_CUDA)
    if (i > 0)
    {
        tmplist[n++] = new ShapeList(list, i);
    }

    return new ShapeList(tmplist, n);
#else
    return mesh;
#endif
}
//
// ----------------------------------------------------------------------------
//

constexpr inline uint32_t make_id(char a, char b, char c, char d) { return a | b << 8 | c << 16 | d << 24; }

#if !defined(USE_CUDA)
Scene load_scene(const char* scn_file, float ratio)
{
    Scene scene;
#else
DScene load_scene(const char* scn_file, float ratio)
{
    DScene scene;
#endif

    if (FILE* fp = fopen(scn_file, "r"))
    {
        static char line[512];

#if !defined(USE_CUDA)
        std::unordered_map<std::string, IMaterial*> materials;
        std::vector<IShape*> objects;
#else
        std::unordered_map<std::string, DMaterial*> materials;
        std::vector<DSphere*> spheres;
        std::vector<DBox*> boxes;
        std::vector<DTriangle*> triangles;
#endif

        while (fgets(line, 512, fp))
        {
            if (line[0] == '#' || line[0] =='\n')
            {
                continue;
            }

            char id[3];
            constexpr uint32_t id_scn = make_id('S', 'C', 'N', '\0');
            constexpr uint32_t id_cam = make_id('C', 'A', 'M', '\0');
            constexpr uint32_t id_mtl = make_id('M', 'T', 'L', '\0');
            constexpr uint32_t id_obj = make_id('O', 'B', 'J', '\0');
            constexpr uint32_t id_tri = make_id('T', 'R', 'I', '\0');

            static char params[512];
            if (sscanf(line, "%c%c%c %[^\n]", &id[0], &id[1], &id[2], params) == 4)
            {
                uint32_t uid = id[0] | id[1] << 8 | id[2] << 16 | 0x0 << 24;
                
                switch (uid)
                {
                case id_scn:
                    fputs("found SCN marker, good!\n", stderr);
                    break;

                case id_cam:
                    fprintf(stderr, "found cam: %s\n", params);
                    {
#if !defined(USE_CUDA)
                        vec3 eye;
                        vec3 center;
                        vec3 up;
#else
                        float3 eye;
                        float3 center;
                        float3 up;
#endif
                        float fov;

                        if (sscanf(params, "(%f,%f,%f) (%f,%f,%f) (%f,%f,%f) %f", &eye.x, &eye.y, &eye.z,
                                                                                  &center.x, &center.y, &center.z,
                                                                                  &up.x, &up.y, &up.z,
                                                                                  &fov) == 10)
                        {
#if !defined(USE_CUDA)
                            scene.cam.setup(eye, center, up, fov, ratio);
#else
                            scene.cam = *camera_create(eye, center, up, fov, ratio);
#endif
                        }
                    }
                    break;

                case id_mtl:
                    fprintf(stderr, "found mtl: %s\n", params);
                    {
                        char mat_name[16];
                        char mat_type;
#if !defined(USE_CUDA)
                        vec3 albedo;
#else
                        float3 albedo;
                        DMaterial::Type type;
#endif
                        float param = .0f;

                        int num = sscanf(params, "%s %c (%f,%f,%f) %f", mat_name,
                                                                        &mat_type,
                                                                        &albedo.x, &albedo.y, &albedo.z,
                                                                        &param);
                        {
                            switch (mat_type)
                            {
#if !defined(USE_CUDA)
                            case 'E':
                                materials[mat_name] = new Emissive(new Constant(albedo));
                                break;
                            case 'L':
                                materials[mat_name] = new Lambertian(new Constant(albedo));
                                break;
                            case 'M':
                                materials[mat_name] = new Metal(albedo, (num == 6) ? param : .0f);
                                break;
                            case 'D':
                                materials[mat_name] = new Dielectric(albedo, (num == 6) ? param : 1.0f);
                                break;
#else
                            case 'E': type = DMaterial::eEMISSIVE; break;
                            case 'L': type = DMaterial::eLAMBERTIAN; break;
                            case 'M': type = DMaterial::eMETAL; break;
                            case 'D': type = DMaterial::eDIELECTRIC; break;
#endif
                            }

#if defined(USE_CUDA)
                            if (type != DMaterial::MAX_TYPES)
                            {
                                materials[mat_name] = create_material(type, albedo, (num == 6) ? param : .0f, (num == 6) ? param : 1.f);
                            }
#endif
                        }
                    }
                    break;

                case id_obj:
                    fprintf(stderr, "found obj: %s\n", params);
                    {
                        char obj_type;
                        char subparams[64];
                        if (sscanf(params, "%c %[^\n]", &obj_type, subparams) == 2)
                        {
                            switch (obj_type)
                            {
                            case 'S':
                            {
#if !defined(USE_CUDA)
                                vec3 center;
#else
                                float3 center;
#endif
                                float radius;
                                char mat_name[16];

                                if (sscanf(subparams, "(%f,%f,%f) %f %s", &center.x, &center.y, &center.z, &radius, mat_name) == 5 &&
                                    materials.count(mat_name) > 0)
                                {
#if !defined(USE_CUDA)
                                    objects.push_back(new Sphere(center, radius, materials[mat_name]));
#else
                                    spheres.push_back(sphere_create(center, radius, *materials[mat_name]));
#endif
                                }
                            }
                            break;
                            case 'B':
                            {
#if !defined(USE_CUDA)
                                vec3 min_box;
                                vec3 max_box;
#else
                                float3 min_box;
                                float3 max_box;
#endif
                                char mat_name[16];
                                if (sscanf(subparams, "(%f,%f,%f) (%f,%f,%f) %s", &min_box.x, &min_box.y, &min_box.z,
                                                                                  &max_box.x, &max_box.y, &max_box.z,
                                                                                  mat_name) == 7 &&
                                                                                  materials.count(mat_name) > 0)
                                {
#if !defined(USE_CUDA)
                                    objects.push_back(new Box(min_box, max_box, materials[mat_name]));
#else
                                    boxes.push_back(box_create(min_box, max_box, *materials[mat_name]));
#endif
                                }
                            }
                            break;
                            case 'T':
                            {
#if !defined(USE_CUDA)
                                vec3 v1, v2, v3;
#else
                                float3 v1, v2, v3;
#endif
                                char mat_name[16];
                                if (sscanf(subparams, "(%f,%f,%f) (%f,%f,%f) (%f,%f,%f) %s", &v1.x, &v1.y, &v1.z,
                                                                                             &v2.x, &v2.y, &v2.z,
                                                                                             &v3.x, &v3.y, &v3.z,
                                                                                             mat_name) == 10 &&
                                    materials.count(mat_name) > 0)
                                {
#if !defined(USE_CUDA)
                                    objects.push_back(new Triangle(v1, v2, v3, materials[mat_name]));
#else
                                    triangles.push_back(triangle_create(v1, v2, v3, *materials[mat_name]));
#endif
                                }
                            }
                            break;
                            }
                        }
                    }
                    break;

                case id_tri:
                    {
                        char file_name[256];
                        char mat_name[16];
                        if (sscanf(params, "%s %s", file_name, mat_name) == 2 && materials.count(mat_name) > 0)
                        {
#if !defined(USE_CUDA)
                            objects.push_back(load_mesh(file_name, materials[mat_name]));
#else
                            std::vector<DTriangle*> mesh = load_mesh(file_name, materials[mat_name]);
                            triangles.insert(triangles.end(), mesh.begin(), mesh.end());
#endif
                        }
                    }
                    break;
                }
            }
        }

        fclose(fp);

#if !defined(USE_CUDA)
        IShape** shape_objects = new IShape*[objects.size()];
        int i = 0;
        for (auto object : objects)
        {
            shape_objects[i++] = object;
        }
        scene.world = new ShapeList(shape_objects, i);
#else
        scene.num_triangles = triangles.size();
        scene.h_triangles = new DTriangle*[scene.num_triangles];
        for (int i = 0; i < scene.num_triangles; ++i)
        {
            scene.h_triangles[i] = triangles[i];
        }

        scene.num_boxes = boxes.size();
        scene.h_boxes = new DBox*[scene.num_boxes];
        for (int i = 0; i < scene.num_boxes; ++i)
        {
            scene.h_boxes[i] = boxes[i];
        }

        scene.num_spheres = spheres.size();
        scene.h_spheres = new DSphere*[scene.num_spheres];
        for (int i = 0; i < scene.num_spheres; ++i)
        {
            scene.h_spheres[i] = spheres[i];
        }
#endif
    }

    return scene;
}

