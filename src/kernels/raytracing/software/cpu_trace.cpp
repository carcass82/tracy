/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "cpu_trace.h"
#include "random.h"
#include <cfloat>
#include <ctime>

#if USE_KDTREE
 #include "kdtree.h"
#endif

#if USE_SIMD
 #include "simdhelper.h"
#endif

#if defined(_MSC_VER) && _MSC_VER < 1920
 // pre-vs2019 ms compiler does not support openmp "collapse" clause
 #define collapse(x) 
#endif

namespace
{
	vec3 sqrtf3(const vec3& a)
	{
		return vec3{ sqrtf(a.x), sqrtf(a.y), sqrtf(a.z) };
	}

	vec3 clamp3(const vec3& a, float min, float max)
	{
		return vec3{ clamp(a.x, min, max), clamp(a.y, min, max), clamp(a.z, min, max) };
	}

#if USE_KDTREE
	struct Triangle
	{
		Triangle(const vec3& in_v0, const vec3& in_v1, const vec3& in_v2, uint16_t in_mesh, uint16_t in_triangle)
			: v{ in_v0, in_v1, in_v2 }
			, v0v1(v[1] - v[0])
			, v0v2(v[2] - v[0])
			, mesh_idx(in_mesh)
			, tri_idx(in_triangle)
		{}

		vec3 v[3];
		vec3 v0v1;
		vec3 v0v2;
		uint16_t mesh_idx;
		uint16_t tri_idx;
	};

	struct OptimizedTriangleSOA
	{
		vec3* v0;
		vec3* v0v1;
		vec3* v0v2;
		uint16_t* mesh_idx;
		uint16_t* tri_idx;

		// super hackish and bad on so many levels
		OptimizedTriangleSOA& operator[](size_t i) const { return *const_cast<OptimizedTriangleSOA*>(this); }
	};
#endif
}

//
// CPU Trace - Details
//
struct CpuTrace::CpuTraceDetails
{
	bool IntersectsWithMesh(const Mesh& mesh, const Ray& in_ray, HitData& inout_intersection) const
	{
		bool hit_triangle = false;

		vec3 ray_direction = in_ray.GetDirection();
		vec3 ray_origin = in_ray.GetOrigin();

		int tris = mesh.GetTriCount();
		for (int i = 0; i < tris; ++i)
		{
			const Index i0 = mesh.GetIndex(i * 3);
			const Index i1 = mesh.GetIndex(i * 3 + 1);
			const Index i2 = mesh.GetIndex(i * 3 + 2);
			
			const vec3 v0 = mesh.GetVertex(i0).pos;
			const vec3 v1 = mesh.GetVertex(i1).pos;
			const vec3 v2 = mesh.GetVertex(i2).pos;

			const vec3 v0v1 = v1 - v0;
			const vec3 v0v2 = v2 - v0;

			vec3 pvec = cross(ray_direction, v0v2);
			vec3 tvec = ray_origin - v0;

			float det = dot(v0v1, pvec);
			
			// if the determinant is negative the triangle is backfacing
			// if the determinant is close to 0, the ray misses the triangle
			if (det > EPS)
			{
				float u = dot(tvec, pvec);
				if (u < EPS || u > det)
				{
					continue;
				}

				vec3 qvec = cross(tvec, v0v1);

				float v = dot(ray_direction, qvec);
				if (v < EPS || u + v > det)
				{
					continue;
				}

				float inv_det = rcp(det);
				float t = dot(v0v2, qvec) * inv_det;
				if (t < inout_intersection.t && t > EPS)
				{
					inout_intersection.t = t;
					inout_intersection.uv = vec2{ u, v } * inv_det;
					inout_intersection.triangle_index = i * 3;
					hit_triangle = true;
				}
			}
		}

		return hit_triangle;
	}

	void FillTriangleIntersectionData(const Mesh& mesh, const Ray& in_ray, HitData& inout_intersection) const
	{
		const Vertex v0 = mesh.GetVertex(mesh.GetIndex(inout_intersection.triangle_index + 0));
		const Vertex v1 = mesh.GetVertex(mesh.GetIndex(inout_intersection.triangle_index + 1));
		const Vertex v2 = mesh.GetVertex(mesh.GetIndex(inout_intersection.triangle_index + 2));

		inout_intersection.point = in_ray.GetPoint(inout_intersection.t);
		inout_intersection.normal = (1.f - inout_intersection.uv.x - inout_intersection.uv.y) * v0.normal + inout_intersection.uv.x * v1.normal + inout_intersection.uv.y * v2.normal;
		inout_intersection.uv = (1.f - inout_intersection.uv.x - inout_intersection.uv.y) * v0.uv0 + inout_intersection.uv.x * v1.uv0 + inout_intersection.uv.y * v2.uv0;
		inout_intersection.material = mesh.GetMaterial();
	}

	bool ComputeIntersection(const Scene& scene, const Ray& ray, HitData& intersection_data) const
	{
		bool hit_any_mesh = false;

#if USE_KDTREE

		auto TriangleRayTester = [](const auto* in_triangles, const unsigned in_first, const unsigned in_last, const Ray& in_ray, HitData& intersection_data)
		{

#if USE_SIMD
			bool hit_triangle = false;

			simd_vec3 ray_direction = in_ray.GetDirection();
			simd_vec3 ray_origin = in_ray.GetOrigin();

			static const simd_float simd_EPS(EPS);
			simd_float simd_nearest_t(intersection_data.t);

			for (size_t idx = in_first; idx < in_last; idx += SIMD_WIDTH)
			{
#if USE_AOS
				float valid_triangle[SIMD_WIDTH];
                vec3 v0s[SIMD_WIDTH];
                vec3 v0v1s[SIMD_WIDTH];
                vec3 v0v2s[SIMD_WIDTH];
                for (int i = 0; i < SIMD_WIDTH; ++i)
                {
                    valid_triangle[i] = (idx + i < in_last);
					if (!(idx + i < in_last))
					{
						continue;
					}

                    v0s[i] = in_triangles[idx + i].v0;
                    v0v1s[i] = in_triangles[idx + i].v0v1;
                    v0v2s[i] = in_triangles[idx + i].v0v2;
                }
#else
				float valid_triangle[SIMD_WIDTH];
				for (int i = 0; i < SIMD_WIDTH; ++i)
				{
					valid_triangle[i] = (idx + i < in_last);
				}

				vec3 *v0s = in_triangles->v0 + idx;
				vec3 *v0v1s = in_triangles->v0v1 + idx;
				vec3 *v0v2s = in_triangles->v0v2 + idx;
#endif
				const simd_vec3 v0(v0s);
				const simd_vec3 v0v1(v0v1s);
				const simd_vec3 v0v2(v0v2s);
				const simd_float valid_condition = simd_float(valid_triangle) < simd_ONE;

				simd_vec3 pvec = cross(ray_direction, v0v2);
				
				simd_float det = dot(v0v1, pvec);
				simd_float simd_inv_det = rcp(det);
				
				// if the determinant is negative the triangle is backfacing
				// if the determinant is close to 0, the ray misses the triangle
				simd_float determinant_condition = (det < simd_EPS);
				
				simd_vec3 tvec = ray_origin - v0;
				simd_float simd_u = dot(tvec, pvec);
				simd_float u_condition = ((simd_u < simd_EPS) | (simd_u > det));

				simd_vec3 qvec = cross(tvec, v0v1);
				simd_float simd_v = dot(ray_direction, qvec);
				simd_float v_condition = ((simd_v < simd_EPS) | (simd_u + simd_v > det));

				simd_float t = dot(v0v2, qvec) * simd_inv_det;
				simd_float t_condition = ((t < simd_EPS) | (t >= simd_nearest_t));
				
				simd_float mask = valid_condition | determinant_condition | u_condition | v_condition | t_condition;
				simd_float simd_res = lerp(t, simd_MINUS_ONE, mask);
					
				float* res = simd_res.get_float();
				for (int i = 0; i < SIMD_WIDTH; ++i)
				{
					if (res[i] > EPS && res[i] < intersection_data.t)
					{
						simd_nearest_t = res[i];

						float* u = simd_u.get_float();
                        float* v = simd_v.get_float();
						float* inv_det = simd_inv_det.get_float();

						intersection_data.t = res[i];
						intersection_data.uv = vec2{ u[i], v[i] } * inv_det[i];
#if USE_AOS
						intersection_data.triangle_index = in_triangles[idx + i].tri_idx;
						intersection_data.object_index = in_triangles[idx + i].mesh_idx;
#else
						intersection_data.triangle_index = in_triangles->tri_idx[idx + i];
						intersection_data.object_index = in_triangles->mesh_idx[idx + i];
#endif
						hit_triangle = true;
					}
				}
			}

			return hit_triangle;

#else // USE_SIMD

			bool hit_triangle = false;

			const vec3 ray_direction = in_ray.GetDirection();
			const vec3 ray_origin = in_ray.GetOrigin();

			for (size_t idx = in_first; idx < in_last; ++idx)
			{
#if USE_AOS
				const vec3 v0   = in_triangles[idx].v0;
				const vec3 v0v1 = in_triangles[idx].v0v1;
				const vec3 v0v2 = in_triangles[idx].v0v2;
#else
				const vec3 v0 = in_triangles->v0[idx];
				const vec3 v0v1 = in_triangles->v0v1[idx];
				const vec3 v0v2 = in_triangles->v0v2[idx];
#endif

				vec3 pvec = cross(ray_direction, v0v2);

				float det = dot(v0v1, pvec);

				// if the determinant is negative the triangle is backfacing
				// if the determinant is close to 0, the ray misses the triangle
				if (det > EPS)
				{
					vec3 tvec = ray_origin - v0;
					float u = dot(tvec, pvec);
					if (u < EPS || u > det)
					{
						continue;
					}

					vec3 qvec = cross(tvec, v0v1);
					float v = dot(ray_direction, qvec);
					if (v < EPS || u + v > det)
					{
						continue;
					}

					float inv_det = rcp(det);
					float t = dot(v0v2, qvec) * inv_det;
					if (t > EPS && t < intersection_data.t)
					{
						intersection_data.t = t;
						intersection_data.uv = vec2{ u, v } * inv_det;
#if USE_AOS
						intersection_data.triangle_index = in_triangles[idx].tri_idx;
						intersection_data.object_index = in_triangles[idx].mesh_idx;
#else
						intersection_data.triangle_index = in_triangles->tri_idx[idx];
						intersection_data.object_index = in_triangles->mesh_idx[idx];
#endif
						hit_triangle = true;
					}
				}
			}

			return hit_triangle;

#endif // USE_SIMD
		};




#endif // USE_KDTREE

		for (const Mesh& mesh : scene.GetObjects())
		{
			if (IntersectsWithBoundingBox(mesh.GetAABB(), ray, intersection_data.t))
			{
#if USE_KDTREE
#if USE_AOS
				if (accel::IntersectsWithTree<accel::Tree<OptimizedTriangle>, OptimizedTriangle, TREE_DEPTH + 1>(&SceneTree, ray, intersection_data, TriangleRayTester))
#else
				if (accel::IntersectsWithTree<accel::OptimizedTree<OptimizedTriangleSOA>, OptimizedTriangleSOA>(&SceneTree, ray, intersection_data, TriangleRayTester))
#endif
				{
					hit_any_mesh = true;
					FillTriangleIntersectionData(scene.GetObject(intersection_data.object_index), ray, intersection_data);
				}
				break;
#else
				if (IntersectsWithMesh(mesh, ray, intersection_data))
				{
					hit_any_mesh = true;
					FillTriangleIntersectionData(mesh, ray, intersection_data);
				}
#endif
			}
		}

		return hit_any_mesh;
	}

	void BuildInternalScene(const Scene& scene)
	{
#if USE_KDTREE

		if (scene.GetObjectCount() > UINT16_MAX)
		{
			TracyLog("Unable to represent mesh index\n");
			DEBUG_BREAK();
		}

		BBox scene_bbox{ FLT_MAX, -FLT_MAX };
		vector<Triangle> scene_tris;
		for (uint16_t i = 0; i < scene.GetObjectCount(); ++i)
		{
			const Mesh& mesh = scene.GetObject(i);
			if (mesh.GetTriCount() * 3 > UINT16_MAX)
			{
				TracyLog("Unable to represent triangle index\n");
				DEBUG_BREAK();
			}

			for (uint16_t t = 0; t < mesh.GetTriCount(); ++t)
			{
				uint16_t tri_idx = t * 3;

				const vec3& v0 = mesh.GetVertex(mesh.GetIndex(tri_idx + 0)).pos;
				const vec3& v1 = mesh.GetVertex(mesh.GetIndex(tri_idx + 1)).pos;
				const vec3& v2 = mesh.GetVertex(mesh.GetIndex(tri_idx + 2)).pos;

				scene_bbox.minbound = pmin(mesh.GetAABB().minbound, scene_bbox.minbound);
				scene_bbox.maxbound = pmax(mesh.GetAABB().maxbound, scene_bbox.maxbound);

				scene_tris.emplace_back(v0, v1, v2, i, tri_idx);
			}
		}

		// Triangle-AABB intersection
		auto TriangleAABBTester = [](const Triangle& triangle, const BBox& aabb)
		{
			// triangle - box test using separating axis theorem (https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/pubs/tribox.pdf)
			// code adapted from http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt

			vec3 v0{ triangle.v[0] - aabb.GetCenter() };
			vec3 v1{ triangle.v[1] - aabb.GetCenter() };
			vec3 v2{ triangle.v[2] - aabb.GetCenter() };

			vec3 e0{ v1 - v0 };
			vec3 e1{ v2 - v1 };
			vec3 e2{ v0 - v2 };

			vec3 fe0{ abs(e0.x), abs(e0.y), abs(e0.z) };
			vec3 fe1{ abs(e1.x), abs(e1.y), abs(e1.z) };
			vec3 fe2{ abs(e2.x), abs(e2.y), abs(e2.z) };

			vec3 aabb_hsize = aabb.GetSize() / 2.f;

			auto AxisTester = [](float a, float b, float fa, float fb, float v0_0, float v0_1, float v1_0, float v1_1, float hsize_0, float hsize_1)
			{
				float p0 = a * v0_0 + b * v0_1;
				float p1 = a * v1_0 + b * v1_1;
				
				float rad = fa * hsize_0 + fb * hsize_1;
				return (min(p0, p1) > rad || max(p0, p1) < -rad);
			};

			if (AxisTester( e0.z, -e0.y, fe0.z, fe0.y, v0.y, v0.z, v2.y, v2.z, aabb_hsize.y, aabb_hsize.z) ||
			    AxisTester(-e0.z,  e0.x, fe0.z, fe0.x, v0.x, v0.z, v2.x, v2.z, aabb_hsize.x, aabb_hsize.z) ||
			    AxisTester( e0.y, -e0.x, fe0.y, fe0.x, v1.x, v1.y, v2.x, v2.y, aabb_hsize.x, aabb_hsize.y) ||
			
			    AxisTester( e1.z, -e1.y, fe1.z, fe1.y, v0.y, v0.z, v2.y, v2.z, aabb_hsize.y, aabb_hsize.z) ||
			    AxisTester(-e1.z,  e1.x, fe1.z, fe1.x, v0.x, v0.z, v2.x, v2.z, aabb_hsize.x, aabb_hsize.z) ||
			    AxisTester( e1.y, -e1.x, fe1.y, fe1.x, v0.x, v0.y, v1.x, v1.y, aabb_hsize.x, aabb_hsize.y) ||
			
			    AxisTester( e2.z, -e2.y, fe2.z, fe2.y, v0.y, v0.z, v1.y, v1.z, aabb_hsize.y, aabb_hsize.z) ||
			    AxisTester(-e2.z,  e2.x, fe2.z, fe2.x, v0.x, v0.z, v1.x, v1.z, aabb_hsize.x, aabb_hsize.z) ||
			    AxisTester( e2.y, -e2.x, fe2.y, fe2.x, v1.x, v1.y, v2.x, v2.y, aabb_hsize.x, aabb_hsize.y))
			{
				return false;
			}

			vec3 trimin = pmin(v0, pmin(v1, v2));
			vec3 trimax = pmax(v0, pmax(v1, v2));
			if ((trimin.x > aabb_hsize.x || trimax.x < -aabb_hsize.x) ||
			    (trimin.y > aabb_hsize.y || trimax.y < -aabb_hsize.y) ||
			    (trimin.z > aabb_hsize.z || trimax.z < -aabb_hsize.z))
			{
				return false;
			}
						
			{
				vec3 trinormal = cross(e0, e1);
			
				vec3 vmin, vmax;
				
				if (trinormal.x > .0f) { vmin.x = -aabb_hsize.x - v0.x; vmax.x =  aabb_hsize.x - v0.x; }
				else                   { vmin.x =  aabb_hsize.x - v0.x; vmax.x = -aabb_hsize.x - v0.x; }
			
				if (trinormal.y > .0f) { vmin.y = -aabb_hsize.y - v0.y; vmax.y =  aabb_hsize.y - v0.y; }
				else                   { vmin.y =  aabb_hsize.y - v0.y; vmax.y = -aabb_hsize.y - v0.y; }
			
				if (trinormal.z > .0f) { vmin.z = -aabb_hsize.z - v0.z; vmax.z =  aabb_hsize.z - v0.z; }
				else                   { vmin.z =  aabb_hsize.z - v0.z; vmax.z = -aabb_hsize.z - v0.z; }
			
				if (dot(trinormal, vmin) > .0f || dot(trinormal, vmax) < .0f)
				{
					return false;
				}
			}
			
			return true;
		};

		auto VectorToSOA = [](OptimizedTriangleSOA& trianglelist, const vector<Triangle>& triangles)
		{
			trianglelist.v0 = new vec3[triangles.size()];
			trianglelist.v0v1 = new vec3[triangles.size()];
			trianglelist.v0v2 = new vec3[triangles.size()];
			trianglelist.mesh_idx = new uint16_t[triangles.size()];
			trianglelist.tri_idx = new uint16_t[triangles.size()];

			for (vector<Triangle>::size_type i = 0; i < triangles.size(); ++i)
			{
				trianglelist.v0[i] = triangles[i].v[0];
				trianglelist.v0v1[i] = triangles[i].v0v1;
				trianglelist.v0v2[i] = triangles[i].v0v2;
				trianglelist.mesh_idx[i] = triangles[i].mesh_idx;
				trianglelist.tri_idx[i] = triangles[i].tri_idx;
			}
		};

#if USE_AOS
		SceneTree.elems.reserve(trimesh.size());
		SceneTree.root = accel::BuildTree<Triangle, std::vector>(&SceneTree, scene_tris, scene_bbox, TriangleAABBTester);
#else
		accel::Tree<Triangle> Scene;
		Scene.elems.reserve(scene_tris.size());
		accel::BuildTree<Triangle, std::vector>(&Scene, scene_tris, scene_bbox, TriangleAABBTester);

		SceneTree.root = Scene.root;
		VectorToSOA(SceneTree.elems, Scene.elems);
#endif
	}

#if USE_AOS
	accel::Tree<OptimizedTriangle> SceneTree;
#else
	accel::OptimizedTree<OptimizedTriangleSOA> SceneTree;
#endif

#else
	}
#endif

	//
	// -- platform data for rendering --
	//
	vector<vec3> output;
	uint32_t* bitmap_bytes;
	
#if defined(WIN32)
	HBITMAP bitmap{};
#else
	XImage* bitmap{};
#endif
};


//
// CPU Trace
//

CpuTrace::CpuTrace()
	: details_(new CpuTraceDetails)
{
}

CpuTrace::~CpuTrace()
{
	delete details_;
}

void CpuTrace::Initialize(Handle in_window, int in_width, int in_height, const Scene& in_scene)
{
	win_handle_ = in_window;
	win_width_ = in_width;
	win_height_ = in_height;
	camera_ = &in_scene.GetCamera();
	scene_ = &in_scene;

	details_->output.resize(in_width * in_height);
#if defined(WIN32)
	BITMAPINFO bmi;
	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth = in_width;
	bmi.bmiHeader.biHeight = in_height;
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 32;
	bmi.bmiHeader.biCompression = BI_RGB;
	bmi.bmiHeader.biSizeImage = in_width * in_height * 4;
	HDC hdc = CreateCompatibleDC(GetDC(in_window));
	details_->bitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&details_->bitmap_bytes, nullptr, 0x0);
#else
	details_->bitmap_bytes = new uint32_t[in_width * in_height];
    details_->bitmap = XCreateImage(win_handle_->dpy,
                                  DefaultVisual(win_handle_->dpy, win_handle_->ds),
                                  DefaultDepth(win_handle_->dpy, win_handle_->ds),
                                  ZPixmap,
                                  0,
                                  reinterpret_cast<char*>(details_->bitmap_bytes),
                                  in_width,
                                  in_height,
                                  32,
                                  0);
#endif

	details_->BuildInternalScene(in_scene);
}

void CpuTrace::UpdateScene()
{
	// copy last frame result to bitmap for displaying
	#pragma omp parallel for collapse(2)
	for (int j = 0; j < win_height_; ++j)
	{
		for (int i = 0; i < win_width_; ++i)
		{
			const vec3 float_col = details_->output[j * win_width_ + i];
			const vec3 bitmap_col = clamp3(255.99f * sqrtf3(float_col), .0f, 255.f);
			const uint32_t dst =  (uint8_t)bitmap_col.b       |
			                     ((uint8_t)bitmap_col.g << 8) |
			                     ((uint8_t)bitmap_col.r << 16);

#if defined(WIN32)
			details_->bitmap_bytes[j * win_width_ + i] = dst;
		}
	}
	InvalidateRect(win_handle_, nullptr, FALSE);
	UpdateWindow(win_handle_);

#else

			XPutPixel(details_->bitmap, i, win_height_ - j, dst);
		}
	}
	XPutImage(win_handle_->dpy, win_handle_->win, DefaultGC(win_handle_->dpy, win_handle_->ds), details_->bitmap, 0, 0, 0, 0, win_width_, win_height_);
	XFlush(win_handle_->dpy);
#endif
}

vec3 CpuTrace::Trace(const Ray& ray, uint32_t random_ctx)
{
	Ray current_ray{ ray };
	vec3 current_color{ 1.f, 1.f, 1.f };

	for (int t = 0; t < bounces_; ++t)
	{
		++raycount_;

		HitData intersection_data;
		intersection_data.t = FLT_MAX;

		if (details_->ComputeIntersection(*scene_, current_ray, intersection_data))
		{
#if DEBUG_SHOW_NORMALS
			return .5f * normalize((1.f + mat3(camera_->GetView()) * intersection_data.normal));
#else
			Ray scattered;
			vec3 attenuation;
			vec3 emission;
			if (intersection_data.material->Scatter(current_ray, intersection_data, attenuation, emission, scattered, random_ctx))
			{
				current_color *= attenuation;
				current_ray = scattered;
			}
			else
			{
				current_color *= emission;
				return current_color;
			}
#endif
		}
		else
		{
			Ray dummy_ray;
			vec3 dummy_vec;
			vec3 sky_color;
			scene_->GetSkyMaterial()->Scatter(current_ray, intersection_data, dummy_vec, sky_color, dummy_ray, random_ctx);

			current_color *= sky_color;
			return current_color;
		}
	}

	return {};
}

void CpuTrace::RenderScene()
{
	float frame_counter = (float)camera_->BeginFrame();
	float blend_factor = frame_counter / float(frame_counter + 1);

	#pragma omp parallel for collapse(2) schedule(dynamic, 2)
	for (int j = 0; j < win_height_; ++j)
	{
		for (int i = 0; i < win_width_; ++i)
		{
			static uint32_t random_ctx = 0x12345;
			
			float u = (i + fastrand(random_ctx)) / float(win_width_);
			float v = (j + fastrand(random_ctx)) / float(win_height_);

			vec3* pixel = &details_->output[j * win_width_ + i];

			vec3 current_color = *pixel;
			vec3 traced_color = Trace(camera_->GetRayFrom(u, v), random_ctx);

			*pixel = lerp(traced_color, current_color, blend_factor);
		}
	}
	camera_->EndFrame();
}

void CpuTrace::OnPaint()
{
#if defined(WIN32)
	PAINTSTRUCT ps;
	RECT rect;
	HDC hdc = BeginPaint(win_handle_, &ps);
	GetClientRect(win_handle_, &rect);

	HDC srcDC = CreateCompatibleDC(hdc);
	SetStretchBltMode(hdc, COLORONCOLOR);
	SelectObject(srcDC, details_->bitmap);
	StretchBlt(hdc, 0, 0, rect.right, rect.bottom, srcDC, 0, 0, win_width_, win_height_, SRCCOPY);
	DeleteObject(srcDC);

	EndPaint(win_handle_, &ps);
#endif
}
