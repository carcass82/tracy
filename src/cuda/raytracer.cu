/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include <cstdio>
#include <cfloat>
#include <cstdint>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "mathutils.cuh"


#if !defined(BUILD_GUI) || defined(_DEBUG)
#define CUDALOG(...) printf(__VA_ARGS__)
#else 
#define CUDALOG(...) do {} while(0);
#endif

//
// from helper_cuda.h
// NVidia CUDA samples
// 
template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        cudaError_t cuda_error = cudaGetLastError();

        CUDALOG("[CUDA error] at %s:%d code=%d (%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorName(cuda_error), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void ensure(cudaError_t val, const char *const file, int const line)
{
    if (val != cudaSuccess)
    {
        CUDALOG("[CUDA error] at %s:%d code=%d (%s)\n", file, line, static_cast<unsigned int>(val), cudaGetErrorName(val));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
#define checkCudaAssert(val) ensure((val), __FILE__, __LINE__)

//
// ----------------------------------------------------------------------------
//

__device__ inline float schlick(float costheta, float ior)
{
    float r0 = (1.f - ior) / (1.f + ior);
    r0 *= r0;
    return r0 + (1.f - r0) * __powf(fmaxf(.0f, (1.f - costheta)), 5);
}

__device__ inline float fastrand(curandState* curand_ctx)
{
    return curand_uniform(curand_ctx);
}

__device__ inline float3 random_on_unit_sphere(curandState* curand_ctx)
{
    float z = fastrand(curand_ctx) * 2.f - 1.f;
    float a = fastrand(curand_ctx) * 2.f * pi();
    float r = __fsqrt_rz(fmaxf(.0f, 1.f - z * z));

	float sin_a;
	float cos_a;
	__sincosf(a, &sin_a, &cos_a);

    return make_float3(r * cos_a, r * sin_a, z);
}

//
// ----------------------------------------------------------------------------
//

struct DRay
{
    float3 origin;
    float3 direction;
};

__device__ inline float3 ray_point_at(const DRay& ray, float t)
{
    return ray.origin + t * ray.direction;
}

struct DMaterial;
struct DIntersection
{
    enum Type { eSPHERE, eBOX, eTRIANGLE, eMESH };

    Type type;
    int index;
    float t;
    float3 point;
    float3 normal;
    float2 uv;
    DMaterial* material;
};

struct DMaterial
{
    enum Type { eLAMBERTIAN, eMETAL, eDIELECTRIC, eISOTROPIC, eEMISSIVE, MAX_TYPES };

    Type type;
    float3 albedo;
    float roughness;
    float ior;
};

__host__ __device__ inline DMaterial* material_create(DMaterial::Type type, float3 albedo, float roughness, float ior)
{
    DMaterial* material = new DMaterial;

    material->type = type;
    material->albedo = albedo;
    material->roughness = roughness;
    material->ior = ior;

    return material;
}

__device__ inline bool material_scatter(DMaterial& material, const DRay& ray, const DIntersection& hit, float3& attenuation, float3& emission, DRay& scattered, curandState* curand_ctx)
{
    switch (material.type)
    {

    case DMaterial::eLAMBERTIAN:
    {
        float3 target = hit.point + hit.normal + random_on_unit_sphere(curand_ctx);
        scattered.origin = hit.point;
        scattered.direction = normalize(target - hit.point);
        attenuation = material.albedo;
        emission = make_float3(.0f, .0f, .0f);

        return true;
    }

    case DMaterial::eMETAL:
    {
        float3 reflected = reflect(ray.direction, hit.normal);
        scattered.origin = hit.point;
        scattered.direction = reflected + material.roughness * random_on_unit_sphere(curand_ctx);

        attenuation = material.albedo;
        emission = make_float3(.0f, .0f, .0f);

        return (dot(scattered.direction, hit.normal) > .0f);
    }

    case DMaterial::eDIELECTRIC:
    {
        float3 outward_normal;
        attenuation = { 1.f, 1.f, 1.f };
        emission = make_float3(.0f, .0f, .0f);

        float ni_nt;
        float cosine;
        if (dot(ray.direction, hit.normal) > .0f)
        {
            outward_normal = -1.f * hit.normal;
            ni_nt = material.ior;
            cosine = dot(ray.direction, hit.normal);
            cosine = __fsqrt_rz(1.f - material.ior * material.ior * (1.f - cosine - cosine));
        }
        else
        {
            outward_normal = hit.normal;
            ni_nt = __frcp_rz(material.ior);
            cosine = -dot(ray.direction, hit.normal);
        }

        float3 refracted;
        bool is_refracted = refract(ray.direction, outward_normal, ni_nt, refracted);
        float reflect_chance = (is_refracted) ? schlick(cosine, material.ior) : 1.0f;

        scattered.origin = hit.point;
        scattered.direction = (fastrand(curand_ctx) < reflect_chance) ? reflect(ray.direction, hit.normal) : refracted;

        return true;
    }

    case DMaterial::eEMISSIVE:
    {
        emission = material.albedo;

        return false;
    }

    default:
        return false;

    };
}

struct DSphere
{
    float3 center;
    float radius;
    DMaterial material;
};

__host__ __device__ inline DSphere* sphere_create(float3 c, float r, DMaterial mat)
{
    DSphere* sphere = new DSphere;
    sphere->center = c;
    sphere->radius = r;
    sphere->material = mat;

    return sphere;
}

__device__ inline float2 sphere_uv(DSphere& sphere, const float3& point)
{
    float phi = atan2f(point.z, point.x);
    float theta = asinf(point.y);

    return make_float2(1.0f - (phi + pi()) / (2.0f * pi()), (theta + pi() / 2.0f) / pi());
}

__device__ inline void sphere_hit_data(DSphere& sphere, const DRay& ray, DIntersection& hit)
{
    hit.point = ray_point_at(ray, hit.t);
    hit.normal = (hit.point - sphere.center) / sphere.radius;
    hit.uv = sphere_uv(sphere, hit.normal);
    hit.material = &sphere.material;
}

struct DBox
{
    float3 min_limit;
    float3 max_limit;
    DMaterial material;
};
    
__host__ __device__ inline DBox* box_create(float3 min, float3 max, DMaterial mat)
{
    DBox* box = new DBox;
    box->min_limit = min;
    box->max_limit = max;
    box->material = mat;

    return box;
}

__device__ inline float3 box_normal(DBox& box, const float3& point)
{
    constexpr float eps = 1e-3f;

    if (fabs(box.min_limit.x - point.x) < eps) return make_float3(-1.f,  .0f,  .0f);
    if (fabs(box.max_limit.x - point.x) < eps) return make_float3( 1.f,  .0f,  .0f);
    if (fabs(box.min_limit.y - point.y) < eps) return make_float3( .0f, -1.f,  .0f);
    if (fabs(box.max_limit.y - point.y) < eps) return make_float3( .0f,  1.f,  .0f);
    if (fabs(box.min_limit.z - point.z) < eps) return make_float3( .0f,  .0f, -1.f);
    return make_float3(.0f, .0f, 1.f);
}

__device__ inline float2 box_uv(DBox& box, const float3& point)
{
    constexpr float eps = 1e-3f;

    if ((fabsf(box.min_limit.x - point.x) < eps) || (fabsf(box.max_limit.x - point.x) < eps))
    {
        return make_float2((point.y - box.min_limit.y) / (box.max_limit.y - box.min_limit.y), (point.z - box.min_limit.z) / (box.max_limit.z - box.min_limit.z));
    }
    if ((fabsf(box.min_limit.y - point.y) < eps) || (fabsf(box.max_limit.y - point.y) < eps))
    {
        return make_float2((point.x - box.min_limit.x) / (box.max_limit.x - box.min_limit.x), (point.z - box.min_limit.z) / (box.max_limit.z - box.min_limit.z));
    }
    return make_float2((point.x - box.min_limit.x) / (box.max_limit.x - box.min_limit.x), (point.y - box.min_limit.y) / (box.max_limit.y - box.min_limit.y));
}

__device__ inline void box_hit_data(DBox& box, const DRay& ray, DIntersection& hit)
{
    hit.point = ray_point_at(ray, hit.t);
    hit.normal = box_normal(box, hit.point);
    hit.uv = box_uv(box, hit.point);
    hit.material = &box.material;
}

struct DTriangle
{
    float3 vertices[3];
    float3 normal[3];
    float2 uv[3];
    float3 v0v1;
    float3 v0v2;
    DMaterial material;
};

__host__ __device__ inline DTriangle* triangle_create(float3 v1, float3 v2, float3 v3, DMaterial mat)
{
    DTriangle* triangle = new DTriangle;
    triangle->vertices[0] = v1;
    triangle->vertices[1] = v2;
    triangle->vertices[2] = v3;
    triangle->material = mat;

    triangle->v0v1 = v2 - v1;
    triangle->v0v2 = v3 - v1;
    triangle->normal[0] = normalize(cross(triangle->v0v1, triangle->v0v2));
    triangle->normal[1] = normalize(cross(triangle->v0v1, triangle->v0v2));
    triangle->normal[2] = normalize(cross(triangle->v0v1, triangle->v0v2));
    triangle->uv[0] = make_float2( .0f,  .0f);
    triangle->uv[1] = make_float2(1.0f,  .0f);
    triangle->uv[2] = make_float2( .0f, 1.0f);

    return triangle;
}

__host__ __device__ inline DTriangle* triangle_create_with_normals(float3 v1, float3 v2, float3 v3, float3 n1, float3 n2, float3 n3, DMaterial mat)
{
	DTriangle* triangle = triangle_create(v1, v2, v3, mat);
    triangle->normal[0] = n1;
    triangle->normal[1] = n2;
    triangle->normal[2] = n3;

    return triangle;
}

__device__ inline void triangle_hit_data(DTriangle& triangle, const DRay& ray, DIntersection& hit)
{
    hit.point = ray_point_at(ray, hit.t);
    hit.normal = (1.f - hit.uv.x - hit.uv.y) * triangle.normal[0] + hit.uv.x * triangle.normal[1] + hit.uv.y * triangle.normal[2];
    hit.uv = (1.f - hit.uv.x - hit.uv.y) * triangle.uv[0] + hit.uv.x * triangle.uv[1] + hit.uv.y * triangle.uv[2];
    hit.material = &triangle.material;
}

struct DMesh
{
	DBox leafs_bbox[8];
	DTriangle* leafs_triangles[8];
	int leafs_tricount[8];
	
	int tricount;
	DBox bbox;
};

__host__ __device__ inline DMesh* mesh_create(DTriangle* triangles, int num_tris, DMaterial mat)
{
	DMesh* mesh = new DMesh;
	mesh->bbox = *box_create(make_float3(FLT_MAX, FLT_MAX, FLT_MAX), make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX), mat);

	// preallocate 8 * all tris.
	// tricount will be used to copy the right number of triangles on device
	for (int i = 0; i < 8; ++i)
	{
		mesh->leafs_triangles[i] = new DTriangle[num_tris];
		mesh->leafs_tricount[i] = 0;
	}

	//
	// build BVH (WIP, now it just splits triangles in 8 groups)
	//
	for (int i = 0; i < num_tris; ++i)
	{
		mesh->bbox.min_limit = min(mesh->bbox.min_limit, min(triangles[i].vertices[0], min(triangles[i].vertices[1], triangles[i].vertices[2])));
		mesh->bbox.max_limit = max(mesh->bbox.max_limit, max(triangles[i].vertices[0], max(triangles[i].vertices[1], triangles[i].vertices[2])));
	}

#if defined(DEBUG_BVH)
	DMaterial debug_materials[] = {
		{ DMaterial::eLAMBERTIAN, make_float3(1.f, 0.f, 0.f), 1.f, 0.f },
		{ DMaterial::eLAMBERTIAN, make_float3(0.f, 1.f, 0.f), 1.f, 0.f },
		{ DMaterial::eLAMBERTIAN, make_float3(0.f, 0.f, 1.f), 1.f, 0.f },
		{ DMaterial::eLAMBERTIAN, make_float3(1.f, 0.f, 1.f), 1.f, 0.f },
		{ DMaterial::eLAMBERTIAN, make_float3(0.f, 1.f, 1.f), 1.f, 0.f },
		{ DMaterial::eLAMBERTIAN, make_float3(1.f, 1.f, 0.f), 1.f, 0.f },
		{ DMaterial::eLAMBERTIAN, make_float3(1.f, 1.f, 1.f), 1.f, 0.f },
		{ DMaterial::eLAMBERTIAN, make_float3(.1f, .1f, .1f), 1.f, 0.f }
	};
#endif

	mesh->tricount = 0;
	{
		float3 cur_minbound = mesh->bbox.min_limit;
		float3 cur_center   = (mesh->bbox.min_limit + mesh->bbox.max_limit) / 2.f;
		float3 cur_maxbound = mesh->bbox.max_limit;

#if defined(DEBUG_BVH)
		mesh->leafs_bbox[0] = *box_create(cur_minbound,                                              cur_center,                                                debug_materials[0]);
		mesh->leafs_bbox[1] = *box_create(make_float3(cur_minbound.x, cur_minbound.y, cur_center.z), make_float3(cur_center.x, cur_center.y, cur_maxbound.z),   debug_materials[1]);
		mesh->leafs_bbox[2] = *box_create(make_float3(cur_center.x, cur_minbound.y, cur_minbound.z), make_float3(cur_maxbound.x, cur_center.y, cur_center.z),   debug_materials[2]);
		mesh->leafs_bbox[3] = *box_create(make_float3(cur_center.x, cur_minbound.y, cur_center.z),   make_float3(cur_maxbound.x, cur_center.y, cur_maxbound.z), debug_materials[3]);
		
		mesh->leafs_bbox[4] = *box_create(make_float3(cur_minbound.x, cur_center.y, cur_minbound.z), make_float3(cur_center.x, cur_maxbound.y, cur_center.z),   debug_materials[4]);
		mesh->leafs_bbox[5] = *box_create(make_float3(cur_minbound.x, cur_center.y, cur_center.z),   make_float3(cur_center.x, cur_maxbound.y, cur_maxbound.z), debug_materials[5]);
		mesh->leafs_bbox[6] = *box_create(make_float3(cur_center.x, cur_center.y, cur_minbound.z),   make_float3(cur_maxbound.x, cur_maxbound.y, cur_center.z), debug_materials[6]);
		mesh->leafs_bbox[7] = *box_create(cur_center,                                                cur_maxbound,                                              debug_materials[7]);
#else
		mesh->leafs_bbox[0] = *box_create(cur_minbound, cur_center, mat);
		mesh->leafs_bbox[1] = *box_create(make_float3(cur_minbound.x, cur_minbound.y, cur_center.z), make_float3(cur_center.x, cur_center.y, cur_maxbound.z), mat);
		mesh->leafs_bbox[2] = *box_create(make_float3(cur_center.x, cur_minbound.y, cur_minbound.z), make_float3(cur_maxbound.x, cur_center.y, cur_center.z), mat);
		mesh->leafs_bbox[3] = *box_create(make_float3(cur_center.x, cur_minbound.y, cur_center.z), make_float3(cur_maxbound.x, cur_center.y, cur_maxbound.z), mat);

		mesh->leafs_bbox[4] = *box_create(make_float3(cur_minbound.x, cur_center.y, cur_minbound.z), make_float3(cur_center.x, cur_maxbound.y, cur_center.z), mat);
		mesh->leafs_bbox[5] = *box_create(make_float3(cur_minbound.x, cur_center.y, cur_center.z), make_float3(cur_center.x, cur_maxbound.y, cur_maxbound.z), mat);
		mesh->leafs_bbox[6] = *box_create(make_float3(cur_center.x, cur_center.y, cur_minbound.z), make_float3(cur_maxbound.x, cur_maxbound.y, cur_center.z), mat);
		mesh->leafs_bbox[7] = *box_create(cur_center, cur_maxbound, mat);
#endif

		for (int i = 0; i < num_tris; ++i)
		{
			DTriangle* triangle = &triangles[i];
			float3 barycenter = (triangle->vertices[0] + triangle->vertices[1] + triangle->vertices[2]) / 3.f;

			if (barycenter.y >= cur_minbound.y && barycenter.y <= cur_center.y)
			{
				if (barycenter.x >= cur_minbound.x && barycenter.x <= cur_center.x &&
					barycenter.z >= cur_minbound.z && barycenter.z <= cur_center.z)
				{
					mesh->leafs_triangles[0][mesh->leafs_tricount[0]++] = *triangle;
					mesh->tricount++;
				}
				else if (barycenter.x >= cur_minbound.x && barycenter.x <= cur_center.x &&
					     barycenter.z >= cur_center.z && barycenter.z <= cur_maxbound.z)
				{
					mesh->leafs_triangles[1][mesh->leafs_tricount[1]++] = *triangle;
					mesh->tricount++;
				}
				else if (barycenter.x >= cur_center.x && barycenter.x <= cur_maxbound.x &&
					     barycenter.z >= cur_minbound.z && barycenter.z <= cur_center.z)
				{
					mesh->leafs_triangles[2][mesh->leafs_tricount[2]++] = *triangle;
					mesh->tricount++;
				}
				else if (barycenter.x >= cur_center.x && barycenter.x <= cur_maxbound.x &&
					     barycenter.z >= cur_center.z && barycenter.z <= cur_maxbound.z)
				{
					mesh->leafs_triangles[3][mesh->leafs_tricount[3]++] = *triangle;
					mesh->tricount++;
				}
			}
			else if (barycenter.y >= cur_center.y && barycenter.y <= cur_maxbound.y)
			{
				if (barycenter.x >= cur_minbound.x && barycenter.x <= cur_center.x &&
					barycenter.z >= cur_minbound.z && barycenter.z <= cur_center.z)
				{
					mesh->leafs_triangles[4][mesh->leafs_tricount[4]++] = *triangle;
					mesh->tricount++;
				}
				else if (barycenter.x >= cur_minbound.x && barycenter.x <= cur_center.x &&
					     barycenter.z >= cur_center.z && barycenter.z <= cur_maxbound.z)
				{
					mesh->leafs_triangles[5][mesh->leafs_tricount[5]++] = *triangle;
					mesh->tricount++;
				}
				else if (barycenter.x >= cur_center.x && barycenter.x <= cur_maxbound.x &&
					     barycenter.z >= cur_minbound.z && barycenter.z <= cur_center.z)
				{
					mesh->leafs_triangles[6][mesh->leafs_tricount[6]++] = *triangle;
					mesh->tricount++;
				}
				else if (barycenter.x >= cur_center.x && barycenter.x <= cur_maxbound.x &&
					     barycenter.z >= cur_center.z && barycenter.z <= cur_maxbound.z)
				{
					mesh->leafs_triangles[7][mesh->leafs_tricount[7]++] = *triangle;
					mesh->tricount++;
				}
			}
		}
	}

	return mesh;
}

__device__ inline void mesh_hit_data(DMesh& mesh, const DRay& ray, DIntersection& hit)
{
	return;
}

struct DCamera
{
    float3 pos;
    float3 horizontal;
    float3 vertical;
    float3 origin;
	
	//float3 position;
	//mat4 view;
	//mat4 projection;
	//mat4 view_projection_inv;
};

__host__ __device__ inline DCamera* camera_create(const float3& eye, const float3& center, const float3& up, float fov, float ratio)
{
    DCamera* camera = new DCamera;

    float theta = radians(fov);
    float height_2 = tanf(theta / 2.f);
    float width_2 = height_2 * ratio;
	
    float3 w = normalize(eye - center);
    float3 u = normalize(cross(up, w));
    float3 v = cross(w, u);
	
    camera->pos = eye;
    camera->horizontal = 2.f * width_2 * u;
    camera->vertical = 2.f * height_2 * v;
    camera->origin = eye - width_2 * u - height_2 * v - w;

	//camera->position = eye;
	//
	//camera->view = lookAt(eye, center, up);
	//camera->projection = perspective(radians(fov), ratio, .1f, 1000.f);
	//
	//camera->view_projection_inv = inverse(camera->projection * camera->view);

    return camera;
}

__device__ inline DRay camera_get_ray(const DCamera& camera, float s, float t)
{
    DRay ray;

    ray.origin = camera.pos;
    ray.direction = normalize(camera.origin + s * camera.horizontal + t * camera.vertical - camera.pos);

	//float3 pixel_ndc = make_float3(s, t, 1.f) * 2.f - 1.f;
	//
	//float4 point_3d = camera.view_projection_inv * make_float4(pixel_ndc.x, pixel_ndc.y, pixel_ndc.z, 1.f);
	//point_3d /= point_3d.w;
	//
	//ray.origin = camera.position;
	//ray.direction = make_float3(point_3d.x, point_3d.y, point_3d.z) - camera.position;

    return ray;
}

__device__ bool intersect_spheres(const DRay& ray, const DSphere* spheres, int sphere_count, DIntersection& hit_data)
{
    bool hit_something = false;

    for (int i = 0; i < sphere_count; ++i)
    {
        const DSphere& sphere = spheres[i];

        const float3 oc = ray.origin - sphere.center;
        const float b = dot(oc, ray.direction);
        const float c = dot(oc, oc) - sphere.radius * sphere.radius;

        if (b <= .0f || c <= .0f)
        {
            const float discriminant = __fsqrt_rz(b * b - c);
            if (discriminant > 0.f)
            {
                const float t0 = -b - discriminant;
                if (t0 > 1e-2f && t0 < hit_data.t)
                {
                    hit_data.t = t0;
                    hit_data.type = DIntersection::eSPHERE;
                    hit_data.index = i;
                    hit_something = true;
                }

                const float t1 = -b + discriminant;
                if (t1 > 1e-2f && t1 < hit_data.t)
                {
                    hit_data.t = t1;
                    hit_data.type = DIntersection::eSPHERE;
                    hit_data.index = i;
                    hit_something = true;
                }
            }
        }
    }

    return hit_something;
}

__device__ bool intersect_boxes(const DRay& ray, const DBox* boxes, int box_count, DIntersection& hit_data)
{
    bool hit_something = false;

    const float3 inv_ray = 1.f / ray.direction;

    for (int i = 0; i < box_count; ++i)
    {
        const DBox& box = boxes[i];

        float tmin = 1e-2f;
        float tmax = hit_data.t;

        bool boxhit = false;

        const float3 minbound = (box.min_limit - ray.origin) * inv_ray;
        const float3 maxbound = (box.max_limit - ray.origin) * inv_ray;
        const float minb[] = { minbound.x, minbound.y, minbound.z };
        const float maxb[] = { maxbound.x, maxbound.y, maxbound.z };

        #pragma unroll
        for (int side = 0; side < 3; ++side)
        {
            float t1 = minb[side];
            float t2 = maxb[side];

            tmin = fmaxf(tmin, fminf(t1, t2));
            tmax = fminf(tmax, fmaxf(t1, t2));

            if (tmin > tmax || tmin > hit_data.t)
            {
                boxhit = false;
                break;
            }
            boxhit = true;
        }

        if (boxhit)
        {
            hit_data.t = tmin;
            hit_data.type = DIntersection::eBOX;
            hit_data.index = i;
            hit_something = true;
        }
    }

    return hit_something;
}

__device__ bool intersect_triangles(const DRay& ray, const DTriangle* triangles, int triangle_count, DIntersection& hit_data)
{
    bool hit_something = false;

    for (int i = 0; i < triangle_count; ++i)
    {
        const DTriangle& triangle = triangles[i];
        {
            float3 pvec = cross(ray.direction, triangle.v0v2);
            float det = dot(triangle.v0v1, pvec);
    
            // if the determinant is negative the triangle is backfacing
            // if the determinant is close to 0, the ray misses the triangle
            if (det < 1e-6f)
            {
                continue;
            }
    
            float inv_det = __frcp_rz(det);
    
            float3 tvec = ray.origin - triangle.vertices[0];
            float u = dot(tvec, pvec) * inv_det;
            if (u < .0f || u > 1.f)
            {
                continue;
            }
    
            float3 qvec = cross(tvec, triangle.v0v1);
            float v = dot(ray.direction, qvec) * inv_det;
            if (v < .0f || u + v > 1.f)
            {
                continue;
            }
    
            float t = dot(triangle.v0v2, qvec) * inv_det;
            if (t < hit_data.t)
            {
                hit_data.t = t;
                hit_data.uv = make_float2(u, v);
                hit_data.type = DIntersection::eTRIANGLE;
                hit_data.index = i;
                hit_something = true;
            }
        }
    }

    return hit_something;
}

__device__ bool intersect_meshes(const DRay& ray, const DMesh* meshes, int mesh_count, DIntersection& hit_data)
{
	bool hit_something = false;

	for (int i = 0; i < mesh_count; ++i)
	{
		const DMesh& mesh = meshes[i];
		{
			DIntersection bvh_hitdata = hit_data;
			hit_something = intersect_boxes(ray, &mesh.bbox, 1, bvh_hitdata);
			if (hit_something)
			{
				hit_something = intersect_boxes(ray, mesh.leafs_bbox, 8, bvh_hitdata);

				if (hit_something)
				{
#if defined(DEBUG_BVH)
					// DEBUG - show BVH boxes
					hit_data = bvh_hitdata;
					box_hit_data(*const_cast<DBox*>(&mesh.leafs_bbox[bvh_hitdata.index]), ray, hit_data);

					hit_data.index = i;
					hit_data.type = DIntersection::eMESH;
#else
					int bvh_index = bvh_hitdata.index;
					hit_something = intersect_triangles(ray, mesh.leafs_triangles[bvh_index], mesh.leafs_tricount[bvh_index], hit_data);
					
					if (hit_something)
					{
						triangle_hit_data(mesh.leafs_triangles[bvh_index][hit_data.index], ray, hit_data);
						
						hit_data.index = i;
						hit_data.type = DIntersection::eMESH;
					}
#endif
				}
			}
		}
	}

	return hit_something;
}

__device__ const int MAX_DEPTH = 5;
__device__ const float3 WHITE = {1.f, 1.f, 1.f};
__device__ const float3 BLACK = {0.f, 0.f, 0.f};


__device__ float3 get_color_for(DRay ray, DSphere* spheres, int sphere_count, DBox* boxes, int box_count, DTriangle* triangles, int tri_count, DMesh* meshes, int meshcount, int* raycount, curandState* curand_ctx)
{
    float3 total_color = WHITE;
    DRay current_ray = ray;

    for (int i = 0; i < MAX_DEPTH; ++i)
    {
        //
        // check for hits
        //
        bool hitspheres = false;
        bool hitboxes = false;
        bool hittris = false;
		bool hitmeshes = false;
        DIntersection hit_data;
        hit_data.t = FLT_MAX;

        hitspheres = intersect_spheres(current_ray, spheres, sphere_count, hit_data);
        hitboxes = intersect_boxes(current_ray, boxes, box_count, hit_data);
        hittris = intersect_triangles(current_ray, triangles, tri_count, hit_data);
		hitmeshes = intersect_meshes(current_ray, meshes, meshcount, hit_data);

        ++(*raycount);

        //
        // return color or continue
        //
        if (hitspheres || hitboxes || hittris || hitmeshes)
        {
            if (hit_data.type == DIntersection::eSPHERE)
            {
                sphere_hit_data(spheres[hit_data.index], current_ray, hit_data);
            }
            else if (hit_data.type == DIntersection::eBOX)
            {
                box_hit_data(boxes[hit_data.index], current_ray, hit_data);
            }
            else if (hit_data.type == DIntersection::eTRIANGLE)
            {
                triangle_hit_data(triangles[hit_data.index], current_ray, hit_data);
            }
			else
			{
				mesh_hit_data(meshes[hit_data.index], current_ray, hit_data);
			}

            //
            // debug - show normals
            //
            //return .5f * (1.f + hit_data.normal);

            DRay scattered;
            float3 attenuation;
            float3 emission;
            if (hit_data.material && material_scatter(*hit_data.material, current_ray, hit_data, attenuation, emission, scattered, curand_ctx))
            {
                total_color *= attenuation;
                current_ray = scattered;
            }
            else
            {
                total_color *= emission;
                return total_color;
            }
        }
        else
        {
            return BLACK;
        }
    }

    return BLACK;
}

__global__ void raytrace(int width, int height, int samples, float3* pixels, int* raycount,
	                     DSphere* spheres, int spherecount,
	                     DBox* boxes, int boxcount,
	                     DTriangle* triangles, int tricount,
	                     DMesh* meshes, int meshcount,
	                     DCamera* camera)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    curandState curand_ctx;
    curand_init(clock64(), i, 0, &curand_ctx);
    curandState* local_curand_ctx = &curand_ctx;
  
    //
    // main loop
    //
    int raycount_inst = 0;
    float3 color{ .0f, .0f, .0f };
    for (int sample = 0; sample < samples; ++sample)
    {
        float s = ((i + fastrand(local_curand_ctx)) / static_cast<float>(width));
        float t = ((j + fastrand(local_curand_ctx)) / static_cast<float>(height));

        DRay ray = camera_get_ray(*camera, s, t);
        color += get_color_for(ray, spheres, spherecount, boxes, boxcount, triangles, tricount, meshes, meshcount, &raycount_inst, local_curand_ctx);
    }

    atomicAdd(raycount, raycount_inst);
    
    //
    // debug output if needed
    //
    //color.x = s;
    //color.y = t;
    //color.z = .0f;
    
    if (i < width && j < height)
    {
        float3& pixel = *(&pixels[j * width + i]);
        pixel = color;
    }

    // just to be sure we're running
    if (i == 0 && j == 0 && blockIdx.z == 0) { CUDALOG("[CUDA] running kernel...\n"); }
}

//
// IFace for raytracer.cpp
// 
#include "../scenes.hpp"

constexpr int MAX_GPU = 32;

DScene scene;

struct DCudaData
{
    bool initialized = false;
    int num_gpus = 1;
    int block_threads[MAX_GPU];
    int block_depth[MAX_GPU];
    dim3 dim_block[MAX_GPU];
    dim3 dim_grid[MAX_GPU];

    // scene definition
    DCamera* d_camera[MAX_GPU];

    DSphere* d_spheres[MAX_GPU];
    int num_spheres;

    DBox* d_boxes[MAX_GPU];
    int num_boxes;

    DTriangle* d_triangles[MAX_GPU];
    int num_triangles;

	DMesh* d_meshes[MAX_GPU];
	int num_meshes;

    // output buffer
    float3* d_output_cuda[MAX_GPU];
    float* h_output_cuda[MAX_GPU];

    // raycount stat
    int* d_raycount[MAX_GPU];

#if CUDA_USE_STREAMS
    cudaStream_t d_stream[MAX_GPU];
#endif
};
DCudaData data;

extern "C" void cuda_setup(const char* path, int w, int h)
{
    scene = load_scene(path, float(w) / float(h));
    
    //
    // ---- CUDA init ----
    //
    int num_gpus = 0;
    checkCudaErrors(cudaGetDeviceCount(&num_gpus));
    data.num_gpus = min(num_gpus, MAX_GPU);

    cudaDeviceProp gpu_properties[MAX_GPU];
    for (int i = 0; i < num_gpus; i++)
    {
        checkCudaErrors(cudaGetDeviceProperties(&gpu_properties[i], i));

        CUDALOG("Device %d (%s):\n"
            "\t%d threads\n"
            "\tblocksize: %dx%dx%d\n"
            "\tshmem per block: %lu Kb\n"
            "\tgridsize: %dx%dx%d\n\n",
            i,
            gpu_properties[i].name,
            gpu_properties[i].maxThreadsPerBlock,
            gpu_properties[i].maxThreadsDim[0], gpu_properties[i].maxThreadsDim[1], gpu_properties[i].maxThreadsDim[2],
            static_cast<unsigned long>(gpu_properties[i].sharedMemPerBlock / 1024.f),
            gpu_properties[i].maxGridSize[0], gpu_properties[i].maxGridSize[1], gpu_properties[i].maxGridSize[2]);

        data.block_threads[i] = sqrt(gpu_properties[i].maxThreadsPerBlock) / 2;
        data.block_depth[i] = 1;
        
        data.dim_block[i].x = data.block_threads[i];
        data.dim_block[i].y = data.block_threads[i];
        data.dim_block[i].z = 1;
        
        data.dim_grid[i].x = w / data.dim_block[i].x + 1;
        data.dim_grid[i].y = h / data.dim_block[i].y + 1;
        data.dim_grid[i].z = data.block_depth[i];
    }

#if CUDA_USE_MULTIGPU
    for (int i = data.num_gpus - 1; i >= 0; --i)
    {
#else
    {
        int i = data.num_gpus - 1;
#endif
        checkCudaErrors(cudaSetDevice(i));

#if CUDA_USE_STREAMS
        checkCudaErrors(cudaStreamCreateWithFlags(&data.d_stream[i], cudaStreamNonBlocking));
#endif

        checkCudaErrors(cudaMalloc((void**)&data.d_output_cuda[i], w * h * sizeof(float3)));
        checkCudaErrors(cudaMemset((void*)data.d_output_cuda[i], 0, w * h * sizeof(float3)));
        checkCudaErrors(cudaMallocHost((void**)&data.h_output_cuda[i], w * h * sizeof(float3)));

        checkCudaErrors(cudaMalloc((void**)&data.d_raycount[i], sizeof(int)));

        //
        // ---- scene ----
        //
        checkCudaErrors(cudaMalloc((void**)&data.d_camera[i], sizeof(DCamera)));
        checkCudaErrors(cudaMemcpy(data.d_camera[i], &scene.cam, sizeof(DCamera), cudaMemcpyHostToDevice));

        data.num_spheres = scene.num_spheres;
        if (data.num_spheres > 0)
        {
            checkCudaErrors(cudaMalloc((void**)&data.d_spheres[i], sizeof(DSphere) * data.num_spheres));
            for (int s = 0; s < data.num_spheres; ++s)
            {
                checkCudaErrors(cudaMemcpy(&data.d_spheres[i][s], scene.h_spheres[s], sizeof(DSphere), cudaMemcpyHostToDevice));
            }
        }

        data.num_boxes = scene.num_boxes;
        if (data.num_boxes > 0)
        {
            checkCudaErrors(cudaMalloc((void**)&data.d_boxes[i], sizeof(DBox) * data.num_boxes));
            for (int b = 0; b < data.num_boxes; ++b)
            {
                checkCudaErrors(cudaMemcpy(&data.d_boxes[i][b], scene.h_boxes[b], sizeof(DBox), cudaMemcpyHostToDevice));
            }
        }

		data.num_triangles = scene.num_triangles;
        if (data.num_triangles > 0)
        {
            checkCudaErrors(cudaMalloc((void**)&data.d_triangles[i], sizeof(DTriangle) * data.num_triangles));
            for (int t = 0; t < data.num_triangles; ++t)
            {
                checkCudaErrors(cudaMemcpy(&data.d_triangles[i][t], scene.h_triangles[t], sizeof(DTriangle), cudaMemcpyHostToDevice));
            }
        }

		data.num_meshes = scene.num_meshes;
		if (data.num_meshes > 0)
		{
			checkCudaErrors(cudaMalloc((void**)&data.d_meshes[i], sizeof(DMesh) * data.num_meshes));
			for (int m = 0; m < data.num_meshes; ++m)
			{
				DMesh device_helper;
				memcpy(&device_helper, scene.h_meshes[m], sizeof(DMesh));
				for (int bvh = 0; bvh < 8; ++bvh)
				{
					checkCudaErrors(cudaMalloc((void**)&device_helper.leafs_triangles[bvh], sizeof(DTriangle) * scene.h_meshes[m]->leafs_tricount[bvh]));
					checkCudaErrors(cudaMemcpy(device_helper.leafs_triangles[bvh], scene.h_meshes[m]->leafs_triangles[bvh], sizeof(DTriangle) * scene.h_meshes[m]->leafs_tricount[bvh], cudaMemcpyHostToDevice));
				}

				checkCudaErrors(cudaMemcpy(&data.d_meshes[i][m], &device_helper, sizeof(DMesh), cudaMemcpyHostToDevice));
			}
		}
    }

	scene.clear();

    data.initialized = true;
}

extern "C" void cuda_trace(int w, int h, int ns, float* out_buffer, int& out_raycount)
{
    // ensure output buffer is properly zeroed
    memset(out_buffer, 0, w * h * sizeof(float3));

    CUDALOG("image is %dx%d (%d samples desired)\n", w, h, ns);

    if (!data.initialized)
    {
        CUDALOG("CUDA scene data not initialized, aborting.\n");
        return;
    }

#if CUDA_USE_MULTIGPU
    for (int i = data.num_gpus - 1; i >= 0; --i)
    {
#else
    {
        int i = data.num_gpus - 1;
#endif
        checkCudaErrors(cudaSetDevice(i));

        checkCudaErrors(cudaMemset((void*)data.d_raycount[i], 0, sizeof(int)));

        CUDALOG("raytrace<<<(%d,%d,%d), (%d,%d,%d)>>> on gpu %d\n", data.dim_block[i].x,
                                                                    data.dim_block[i].y,
                                                                    data.dim_block[i].z,
                                                                    data.dim_grid[i].x,
                                                                    data.dim_grid[i].y,
                                                                    data.dim_grid[i].z, i);

#if CUDA_USE_STREAMS
        raytrace<<<data.dim_grid[i], data.dim_block[i], 0, data.d_stream[i]>>>(w,
                                                                               h,
                                                                               ns / data.block_depth[i],
                                                                               data.d_output_cuda[i],
                                                                               data.d_raycount[i],
                                                                               data.d_spheres[i],   data.num_spheres,
                                                                               data.d_boxes[i],     data.num_boxes,
                                                                               data.d_triangles[i], data.num_triangles,
			                                                                   data.d_meshes[i],    data.num_meshes,
                                                                               data.d_camera[i]);

#else
        raytrace<<<data.dim_grid[i], data.dim_block[i] >>>(w,
                                                           h,
                                                           ns / data.block_depth[i],
                                                           data.d_output_cuda[i],
                                                           data.d_raycount[i],
                                                           data.d_spheres[i],   data.num_spheres,
                                                           data.d_boxes[i],     data.num_boxes,
                                                           data.d_triangles[i], data.num_triangles,
			                                               data.d_meshes[i], data.num_meshes,
                                                           data.d_camera[i]);
#endif
        checkCudaAssert(cudaGetLastError());
    }

#if CUDA_USE_MULTIGPU
    for (int i = data.num_gpus - 1; i >= 0; --i)
    {
#else
    {
        int i = data.num_gpus - 1;
#endif
        checkCudaErrors(cudaSetDevice(i));

#if CUDA_USE_STREAMS
        checkCudaErrors(cudaMemcpyAsync(data.h_output_cuda[i], data.d_output_cuda[i], w * h * sizeof(float3), cudaMemcpyDeviceToHost, data.d_stream[i]));

        checkCudaErrors(cudaStreamSynchronize(data.d_stream[i]));
#else
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(data.h_output_cuda[i], data.d_output_cuda[i], w * h * sizeof(float3), cudaMemcpyDeviceToHost));
#endif
      
#if CUDA_USE_MULTIGPU
        int gpu_split = data.num_gpus;

		for (int j = 0; j < w * h * 3; ++j)
        {
            out_buffer[j] += data.h_output_cuda[i][j] / gpu_split;
        }
#else
        memcpy(out_buffer, data.h_output_cuda[i], w * h * 3 * sizeof(float));
#endif

        size_t tmp;
        checkCudaErrors(cudaMemcpy(&tmp, data.d_raycount[i], sizeof(int), cudaMemcpyDeviceToHost));
        out_raycount += tmp;

        CUDALOG("cuda compute (%d/%d) completed!\n", i, data.num_gpus - 1);
    }
}

extern "C" void cuda_cleanup()
{
    if (data.initialized)
    {
#if CUDA_USE_MULTIGPU
        for (int i = data.num_gpus - 1; i >= 0; --i)
        {
#else
        {
            int i = data.num_gpus - 1;
#endif
            checkCudaErrors(cudaSetDevice(i));

            checkCudaErrors(cudaFree(data.d_raycount[i]));
            checkCudaErrors(cudaFree(data.d_output_cuda[i]));
            checkCudaErrors(cudaFreeHost(data.h_output_cuda[i]));

            if (data.num_spheres > 0)
            {
                checkCudaErrors(cudaFree(data.d_spheres[i]));
            }

            if (data.num_boxes > 0)
            {
                checkCudaErrors(cudaFree(data.d_boxes[i]));
            }

            if (data.num_triangles > 0)
            {
                checkCudaErrors(cudaFree(data.d_triangles[i]));
            }

			if (data.num_meshes > 0)
			{
				checkCudaErrors(cudaFree(data.d_meshes[i]));
			}

#if CUDA_USE_STREAMS
            checkCudaErrors(cudaStreamDestroy(data.d_stream[i]));
#endif
        }
    }
}
