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

#include "shapes/box.hpp"
#include "shapes/sphere.hpp"
#include "shapes/triangle.hpp"
#include "shapes/shapelist.hpp"

#include "materials/lambertian.hpp"
#include "materials/metal.hpp"
#include "materials/dielectric.hpp"
#include "materials/emissive.hpp"

#include "textures/bitmap.hpp"
#include "textures/constant.hpp"
#include "textures/checker.hpp"

static IMaterial* debug_materials[] = {
#if defined(DEBUG_BVH)
        new Lambertian(new Constant(vec3{ 1.f, 0.f, 0.f })),
        new Lambertian(new Constant(vec3{ 0.f, 1.f, 0.f })),
        new Lambertian(new Constant(vec3{ 0.f, 0.f, 1.f })),
        new Lambertian(new Constant(vec3{ 1.f, 0.f, 1.f })),
        new Lambertian(new Constant(vec3{ 0.f, 1.f, 1.f })),
        new Lambertian(new Constant(vec3{ 1.f, 1.f, 0.f })),
        new Lambertian(new Constant(vec3{ 1.f, 1.f, 1.f })),
        new Lambertian(new Constant(vec3{ .1f, .1f, .1f }))
#else
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
#endif
};
static constexpr int debug_materials_size = array_size(debug_materials);


struct Scene
{
    Camera cam;
    IShape* world;
};
#else
struct DScene
{
    DCamera cam;

	DMesh** h_meshes;
	int num_meshes;

    DTriangle** h_triangles;
    int num_triangles;

    DBox** h_boxes;
    int num_boxes;

    DSphere** h_spheres;
    int num_spheres;


	void clear()
	{
		for (int i = 0; i < num_meshes; ++i)    { /* TODO: cleanup tris  */ delete h_meshes[i]; }    delete[] h_meshes;
		for (int i = 0; i < num_triangles; ++i) { delete h_triangles[i]; } delete[] h_triangles;
		for (int i = 0; i < num_boxes; ++i)     { delete h_boxes[i]; }     delete[] h_boxes;
		for (int i = 0; i < num_spheres; ++i)   { delete h_spheres[i]; }   delete[] h_spheres;
	}
};
#endif

// NOTE: cubic root of leafcount must be divisible by 8
// default 512 = 8x8x8
#if !defined(USE_CUDA)
IShape* create_bvh(IShape** trimesh, int numtris, int leafcount = 512)
{
    vec3 minbound{  FLT_MAX,  FLT_MAX,  FLT_MAX };
    vec3 maxbound{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (int i = 0; i < numtris; ++i)
    {
        Triangle* triangle = static_cast<Triangle*>(trimesh[i]);

        for (int j = 0; j < 3; ++j)
        {
            minbound = min3(minbound, triangle->get_vertex(j));
            maxbound = max3(maxbound, triangle->get_vertex(j));
        }
    }

    const int leaf_count = leafcount;
    const int cbrt_leafcount = (int)cbrtf((float)leaf_count);
    const int coord_advance = 2;

    int inserted_tris = 0;

    IShape** leafs = new IShape*[leaf_count];
    float slice_size_x = (maxbound.x - minbound.x) / cbrt_leafcount;
    float slice_size_y = (maxbound.y - minbound.y) / cbrt_leafcount;
    float slice_size_z = (maxbound.z - minbound.z) / cbrt_leafcount;
    int i = 0;
    for (int i_x = 0; i_x < cbrt_leafcount; i_x += coord_advance)
    {
        for (int i_y = 0; i_y < cbrt_leafcount; i_y += coord_advance)
        {
            for (int i_z = 0; i_z < cbrt_leafcount; i_z += coord_advance)
            {
                int leaf_low_top_right_tricount = 0;
                int leaf_low_bottom_right_tricount = 0;
                int leaf_low_bottom_left_tricount = 0;
                int leaf_low_top_left_tricount = 0;
                int leaf_high_top_right_tricount = 0;
                int leaf_high_bottom_right_tricount = 0;
                int leaf_high_bottom_left_tricount = 0;
                int leaf_high_top_left_tricount = 0;

                IShape** leaf_low_bottom_right = new IShape*[numtris];
                IShape** leaf_low_top_right = new IShape*[numtris];
                IShape** leaf_low_bottom_left = new IShape*[numtris];
                IShape** leaf_low_top_left = new IShape*[numtris];
                IShape** leaf_high_bottom_right = new IShape*[numtris];
                IShape** leaf_high_top_right = new IShape*[numtris];
                IShape** leaf_high_bottom_left = new IShape*[numtris];
                IShape** leaf_high_top_left = new IShape*[numtris];

                vec3 cur_minbound = minbound + vec3{ (i_x + 0) * slice_size_x, (i_y + 0) * slice_size_y, (i_z + 0) * slice_size_z };
                vec3 cur_center   = minbound + vec3{ (i_x + 1) * slice_size_x, (i_y + 1) * slice_size_y, (i_z + 1) * slice_size_z };
                vec3 cur_maxbound = minbound + vec3{ (i_x + 2) * slice_size_x, (i_y + 2) * slice_size_y, (i_z + 2) * slice_size_z };

                for (int j = 0; j < numtris; ++j)
                {
                    Triangle* triangle = static_cast<Triangle*>(trimesh[j]);
                    const vec3& barycenter = triangle->get_barycenter();

                    //       __________ 
                    //      /----/----/|
                    // min +----+----+||
                    //     |    |    |/|
                    //     +--center-+||
                    //     |    |    |/
                    //     +----+----+ max
                    //

                    if (barycenter.y >= cur_minbound.y && barycenter.y <= cur_center.y)
                    {
                        if (barycenter.x >= cur_minbound.x && barycenter.x <= cur_center.x &&
                            barycenter.z >= cur_minbound.z && barycenter.z <= cur_center.z)
                        {
                            leaf_low_bottom_left[leaf_low_bottom_left_tricount++] = trimesh[j];
                            ++inserted_tris;
                        }
                        else if (barycenter.x >= cur_minbound.x && barycenter.x <= cur_center.x &&
                            barycenter.z >= cur_center.z && barycenter.z <= cur_maxbound.z)
                        {
                            leaf_low_top_left[leaf_low_top_left_tricount++] = trimesh[j];
                            ++inserted_tris;
                        }
                        else if (barycenter.x >= cur_center.x && barycenter.x <= cur_maxbound.x &&
                            barycenter.z >= cur_minbound.z && barycenter.z <= cur_center.z)
                        {
                            leaf_low_bottom_right[leaf_low_bottom_right_tricount++] = trimesh[j];
                            ++inserted_tris;
                        }
                        else if (barycenter.x >= cur_center.x && barycenter.x <= cur_maxbound.x &&
                            barycenter.z >= cur_center.z && barycenter.z <= cur_maxbound.z)
                        {
                            leaf_low_top_right[leaf_low_top_right_tricount++] = trimesh[j];
                            ++inserted_tris;
                        }
                    }
                    else if (barycenter.y >= cur_center.y && barycenter.y <= cur_maxbound.y)
                    {
                        if (barycenter.x >= cur_minbound.x && barycenter.x <= cur_center.x &&
                            barycenter.z >= cur_minbound.z && barycenter.z <= cur_center.z)
                        {
                            leaf_high_bottom_left[leaf_high_bottom_left_tricount++] = trimesh[j];
                            ++inserted_tris;
                        }
                        else if (barycenter.x >= cur_minbound.x && barycenter.x <= cur_center.x &&
                            barycenter.z >= cur_center.z && barycenter.z <= cur_maxbound.z)
                        {
                            leaf_high_top_left[leaf_high_top_left_tricount++] = trimesh[j];
                            ++inserted_tris;
                        }
                        else if (barycenter.x >= cur_center.x && barycenter.x <= cur_maxbound.x &&
                            barycenter.z >= cur_minbound.z && barycenter.z <= cur_center.z)
                        {
                            leaf_high_bottom_right[leaf_high_bottom_right_tricount++] = trimesh[j];
                            ++inserted_tris;
                        }
                        else if (barycenter.x >= cur_center.x && barycenter.x <= cur_maxbound.x &&
                            barycenter.z >= cur_center.z && barycenter.z <= cur_maxbound.z)
                        {
                            leaf_high_top_right[leaf_high_top_right_tricount++] = trimesh[j];
                            ++inserted_tris;
                        }
                    }
                }

#if defined(DEBUG_BVH)
                leafs[8 * i + 0] = new Box(vec3{ cur_minbound.x, cur_minbound.y, cur_minbound.z }, vec3{ cur_center.x, cur_center.y, cur_center.z }, debug_materials[(8 * i + 0) % debug_materials_size]);
                leafs[8 * i + 1] = new Box(vec3{ cur_minbound.x, cur_minbound.y, cur_center.z }, vec3{ cur_center.x, cur_center.y, cur_maxbound.z }, debug_materials[(8 * i + 1) % debug_materials_size]);
                leafs[8 * i + 2] = new Box(vec3{ cur_center.x, cur_minbound.y, cur_minbound.z }, vec3{ cur_maxbound.x, cur_center.y, cur_center.z }, debug_materials[(8 * i + 2) % debug_materials_size]);
                leafs[8 * i + 3] = new Box(vec3{ cur_center.x, cur_minbound.y, cur_center.z }, vec3{ cur_maxbound.x, cur_center.y, cur_maxbound.z }, debug_materials[(8 * i + 3) % debug_materials_size]);
                leafs[8 * i + 4] = new Box(vec3{ cur_minbound.x, cur_center.y, cur_minbound.z }, vec3{ cur_center.x, cur_maxbound.y, cur_center.z }, debug_materials[(8 * i + 4) % debug_materials_size]);
                leafs[8 * i + 5] = new Box(vec3{ cur_minbound.x, cur_center.y, cur_center.z }, vec3{ cur_center.x, cur_maxbound.y, cur_maxbound.z }, debug_materials[(8 * i + 5) % debug_materials_size]);
                leafs[8 * i + 6] = new Box(vec3{ cur_center.x, cur_center.y, cur_minbound.z }, vec3{ cur_maxbound.x, cur_maxbound.y, cur_center.z }, debug_materials[(8 * i + 6) % debug_materials_size]);
                leafs[8 * i + 7] = new Box(vec3{ cur_center.x, cur_center.y, cur_center.z }, vec3{ cur_maxbound.x, cur_maxbound.y, cur_maxbound.z }, debug_materials[(8 * i + 7) % debug_materials_size]);
#else
                leafs[8 * i + 0] = new ShapeList(leaf_low_bottom_left,   leaf_low_bottom_left_tricount);
                leafs[8 * i + 1] = new ShapeList(leaf_low_bottom_right,  leaf_low_bottom_right_tricount);
                leafs[8 * i + 2] = new ShapeList(leaf_low_top_left,      leaf_low_top_left_tricount);
                leafs[8 * i + 3] = new ShapeList(leaf_low_top_right,     leaf_low_top_right_tricount);
                leafs[8 * i + 4] = new ShapeList(leaf_high_bottom_left,  leaf_high_bottom_left_tricount);
                leafs[8 * i + 5] = new ShapeList(leaf_high_bottom_right, leaf_high_bottom_right_tricount);
                leafs[8 * i + 6] = new ShapeList(leaf_high_top_left,     leaf_high_top_left_tricount);
                leafs[8 * i + 7] = new ShapeList(leaf_high_top_right,    leaf_high_top_right_tricount);
#endif
                ++i;
            }
        }
    }

    IShape** prev = leafs;
    int steps = leaf_count / 8;
    while (steps >= 1)
    {
        IShape** cur = new IShape*[steps];
        for (int i = 0; i < steps; ++i)
        {
            cur[i] = new ShapeList(&prev[i * 8], 8);
        }
        steps /= 8;
        prev = cur;
    }

    return prev[0];
}
#endif

#if !defined(USE_CUDA)
IShape* load_mesh(const char* obj_path, IMaterial* obj_material)
{
#else
std::vector<DTriangle> load_mesh(const char* obj_path, DMaterial* obj_material)
{
    std::vector<DTriangle> mesh;
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
    //IShape** tmplist = new IShape*[attrib.vertices.size()];
    //constexpr unsigned MAX_SPLIT = 100;
    //IShape** list = new IShape*[MAX_SPLIT];
    
    IShape** list = new IShape*[attrib.vertices.size()];

    vec3 verts[3];
    vec3 norms[3];
    bool has_normals = true;

    unsigned int i = 0;
#else
    float3 verts[3];
    float3 norms[3];
    bool has_normals = true;
#endif
    unsigned int v = 0;
    for (const tinyobj::shape_t& shape : shapes)
    {
        for (const tinyobj::index_t& index : shape.mesh.indices)
        {
            const float v0 = attrib.vertices[3 * index.vertex_index + 0];
            const float v1 = attrib.vertices[3 * index.vertex_index + 1];
            const float v2 = attrib.vertices[3 * index.vertex_index + 2];

            float n0 = .0f;
            float n1 = .0f;
            float n2 = .0f;
            if (index.normal_index != -1)
            {
                n0 = attrib.normals[3 * index.normal_index + 0];
                n1 = attrib.normals[3 * index.normal_index + 1];
                n2 = attrib.normals[3 * index.normal_index + 2];
            }
            else
            {
                has_normals = false;
            }

#if !defined(USE_CUDA)
            verts[v] = vec3{ v0, v1, v2 };
            norms[v] = vec3{ n0, n1, n2 };
            ++v;

            if (v == 3)
            {
                if (has_normals)
                {
                    list[i++] = new Triangle(verts[0], verts[1], verts[2], norms[0], norms[1], norms[2], obj_material);
                }
                else
                {
                    list[i++] = new Triangle(verts[0], verts[1], verts[2], obj_material);
                }
                v = 0;
            }

            //if (i >= MAX_SPLIT)
            //{
            //    tmplist[n++] = new ShapeList(list, i, debug_materials[n % debug_materials_size]);
            //
            //    i = 0;
            //    list = new IShape*[MAX_SPLIT];
            //}
#else
            verts[v] = make_float3(v0, v1, v2);
            norms[v] = make_float3(n0, n1, n2);
            ++v;

            if (v == 3)
            {
                if (has_normals)
                {
                    mesh.push_back(*triangle_create_with_normals(verts[0], verts[1], verts[2], norms[0], norms[1], norms[2], *obj_material));
                }
                else
                {
                    mesh.push_back(*triangle_create(verts[0], verts[1], verts[2], *obj_material));
                }
                v = 0;
            }
#endif
        }
    }

#if !defined(USE_CUDA)
    //if (i > 0)
    //{
    //    tmplist[n++] = new ShapeList(list, i, debug_materials[n % debug_materials_size]);
    //}
    //
    //return new ShapeList(tmplist, n);

    return create_bvh(list, i);
#else
    return mesh;
#endif
}
//
// ----------------------------------------------------------------------------
//

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
		std::vector<DMesh*> meshes;
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
                                materials[mat_name] = material_create(type, albedo, (num == 6) ? param : .0f, (num == 6) ? param : 1.f);
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
                    fprintf(stderr, "found tri: %s\n", params);
                    {
                        char file_name[256];
                        char mat_name[16];
                        if (sscanf(params, "%s %s", file_name, mat_name) == 2 && materials.count(mat_name) > 0)
                        {
#if !defined(USE_CUDA)
                            objects.push_back(load_mesh(file_name, materials[mat_name]));
#else
                            std::vector<DTriangle> mesh = load_mesh(file_name, materials[mat_name]);
                            meshes.push_back(mesh_create(&mesh[0], mesh.size(), *materials[mat_name]));
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
		scene.num_meshes = meshes.size();
		scene.h_meshes = new DMesh*[scene.num_meshes];
		for (int i = 0; i < scene.num_meshes; ++i)
		{
			scene.h_meshes[i] = meshes[i];
		}

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

