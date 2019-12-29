/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "cpu_trace.h"
#include "random.h"
#include "kdtree.h"
#include <cfloat>
#include <ctime>

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
}

//
// CPU Trace - Details
//
struct CpuTrace::CpuTraceDetails
{
#if USE_KDTREE
	bool IntersectsWithMesh(const vector<accel::Triangle>& mesh, const Ray& in_ray, HitData& inout_intersection) const
#else
	bool IntersectsWithMesh(const Mesh& mesh, const Ray& in_ray, HitData& inout_intersection) const
#endif
	{
		bool hit_triangle = false;

		vec3 ray_direction = in_ray.GetDirection();
		vec3 ray_origin = in_ray.GetOrigin();

#if USE_KDTREE
		for (auto&& tri : mesh)
		{
			const vec3 v0 = tri.vertices[0];

			const vec3 v0v1 = tri.v0v1;
			const vec3 v0v2 = tri.v0v2;
#else
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
#endif

			vec3 pvec = cross(ray_direction, v0v2);
			vec3 tvec = ray_origin - v0;

			float det = dot(v0v1, pvec);
			float inv_det = rcp(det);

			// if the determinant is negative the triangle is backfacing
			// if the determinant is close to 0, the ray misses the triangle
			if (det > 1.e-8f)
			{
				float u = dot(tvec, pvec);
				if (u < .0f || u > det)
				{
					continue;
				}

				vec3 qvec = cross(tvec, v0v1);

				float v = dot(ray_direction, qvec);
				if (v < .0f || u + v > det)
				{
					continue;
				}

				float t = dot(v0v2, qvec) * inv_det;
				if (t < inout_intersection.t && t > 1.e-3f)
				{
					inout_intersection.t = t;
					inout_intersection.uv = vec2{ u, v } * inv_det;
#if USE_KDTREE
					inout_intersection.triangle_index = tri.tri_idx;
					inout_intersection.object_index = tri.mesh_idx;
#else
					inout_intersection.triangle_index = i * 3;
#endif
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

		if (accel::IntersectsWithTree<accel::Triangle>(SceneTree,
		                                               ray,
		                                               intersection_data,
		                                               [=](const vector<accel::Triangle> t, const Ray& r, HitData& h) { return IntersectsWithMesh(t, r, h); }))
		{
			hit_any_mesh = true;
			FillTriangleIntersectionData(scene.GetObject(intersection_data.object_index),
			                             ray,
			                             intersection_data);
		}

#else
		
		for (const Mesh& mesh : scene.GetObjects())
		{
			if (IntersectsWithBoundingBox(mesh.GetAABB(), ray, intersection_data.t))
			{
				if (IntersectsWithMesh(mesh, ray, intersection_data))
				{
					hit_any_mesh = true;
					FillTriangleIntersectionData(mesh, ray, intersection_data);
				}
			}
		}

#endif

		return hit_any_mesh;
	}

	void BuildInternalScene(const Scene& scene)
	{
#if USE_KDTREE
		BBox root{ FLT_MAX, -FLT_MAX };
		vector<const accel::Triangle*> trimesh;
		for (int i = 0; i < scene.GetObjectCount(); ++i)
		{
			const Mesh& mesh = scene.GetObject(i);

			for (int t = 0; t < mesh.GetTriCount(); ++t)
			{
				int tri_idx = t * 3;

				const vec3& v0 = mesh.GetVertex(mesh.GetIndex(tri_idx + 0)).pos;
				const vec3& v1 = mesh.GetVertex(mesh.GetIndex(tri_idx + 1)).pos;
				const vec3& v2 = mesh.GetVertex(mesh.GetIndex(tri_idx + 2)).pos;

				root.minbound = pmin(mesh.GetAABB().minbound, root.minbound);
				root.maxbound = pmax(mesh.GetAABB().maxbound, root.maxbound);

				trimesh.emplace_back(new accel::Triangle{ v0, v1, v2, i, tri_idx });
			}
		}

		// Triangle-AABB intersection
		auto TriangleAABBTester = [](const accel::Triangle& triangle, const BBox& aabb)
		{
			// triangle - box test using separating axis theorem (https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/pubs/tribox.pdf)
			// code adapted from http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt

			vec3 v0{ triangle.vertices[0] - aabb.GetCenter() };
			vec3 v1{ triangle.vertices[1] - aabb.GetCenter() };
			vec3 v2{ triangle.vertices[2] - aabb.GetCenter() };

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

		SceneTree = *accel::BuildTree<accel::Triangle, 200, 15>(trimesh, root, TriangleAABBTester);
	}

	accel::Tree<accel::Triangle> SceneTree;
#else
	}
#endif

	uint32_t randomctx = 0;

	//
	// -- platform data for rendering --
	//
	vector<vec3> output;
	uint32_t* bitmap_bytes;
	
#if defined(WIN32)
	HBITMAP bitmap;
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
	details_->randomctx = static_cast<uint32_t>(time(nullptr));
	fastrand(details_->randomctx);
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
	#pragma omp parallel for
	for (int j = 0; j < win_height_; ++j)
	{
		for (int i = 0; i < win_width_; ++i)
		{
			const vec3 bitmap_col = clamp3(255.99f * sqrtf3(details_->output[j * win_width_ + i]), .0f, 255.f);
			const uint32_t dst = (uint8_t)bitmap_col.b |
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
	float blend_factor = frame_counter_ / float(frame_counter_ + 1);

#if defined(_MSC_VER) && _MSC_VER < 1920
 // ms compiler does not support openmp 3.0 until vs2019
 // (where it must be enabled with /openmp:experimental switch)
 #define collapse(x) 
#endif

	uint32_t random_ctx = details_->randomctx;

	#pragma omp parallel for collapse(3) schedule(dynamic) private(random_ctx)
	for (int j = 0; j < win_height_; ++j)
	{
		for (int i = 0; i < win_width_; ++i)
		{
			for (int s = 0; s < samples_; ++s)
			{
				float u = (i + fastrand(random_ctx)) / float(win_width_);
				float v = (j + fastrand(random_ctx)) / float(win_height_);
				
				vec3 sampled_col = Trace(camera_->GetRayFrom(u, v), random_ctx);
				sampled_col /= (float)samples_;

				#pragma omp critical
				{
					vec3 old_col = details_->output[j * win_width_ + i];
					details_->output[j * win_width_ + i] = lerp(sampled_col, old_col, blend_factor);
				}
			}
		}
	}

	details_->randomctx = random_ctx;
	++frame_counter_;
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
