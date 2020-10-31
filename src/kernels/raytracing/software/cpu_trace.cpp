/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "cpu_trace.h"
#include "random.h"
#include "collision.h"
#include <cfloat>
#include <ctime>

#if USE_KDTREE
 #include "kdtree.h"
#endif

#if defined(_MSC_VER) && _MSC_VER < 1920
 // pre-vs2019 ms compiler does not support openmp "collapse" clause
 #define collapse(x) 
#endif

namespace
{
#if USE_KDTREE
	struct Tri
	{
		constexpr Tri()
		{}

		constexpr Tri(uint32_t mesh_idx, uint32_t triangle_idx, const vec3& v0, const vec3& v1, const vec3& v2)
			: packed_tri_info((mesh_idx << 24) | triangle_idx), vertices{v0, v1, v2}
		{}

		constexpr uint32_t GetMeshId() const     { return packed_tri_info >> 24; }
		constexpr uint32_t GetTriangleId() const { return packed_tri_info & 0xffffff; }

		uint32_t packed_tri_info{ 0 };
		vec3 vertices[3]{};
	};
#endif
}

//
// CPU Trace - Details
//
struct CpuTrace::CpuTraceDetails
{
	void FillTriangleIntersectionData(const Mesh& mesh, const Ray& in_ray, HitData& inout_intersection) const
	{
		const Vertex v0 = mesh.GetVertex(mesh.GetIndex(inout_intersection.triangle_index + 0));
		const Vertex v1 = mesh.GetVertex(mesh.GetIndex(inout_intersection.triangle_index + 1));
		const Vertex v2 = mesh.GetVertex(mesh.GetIndex(inout_intersection.triangle_index + 2));
		const vec2 uv = inout_intersection.uv;
		const Material* material = mesh.GetMaterial();

		inout_intersection.point = in_ray.GetPoint(inout_intersection.t);
		inout_intersection.normal = (1.f - uv.x - uv.y) * v0.normal + uv.x * v1.normal + uv.y * v2.normal;
		inout_intersection.uv = (1.f - uv.x - uv.y) * v0.uv0 + uv.x * v1.uv0 + uv.y * v2.uv0;
		inout_intersection.material = material;
	}

	bool ComputeIntersection(const Scene& scene, const Ray& ray, HitData& intersection_data) const
	{
		bool hit_any_mesh = false;

#if USE_KDTREE

		auto TriangleRayTester = [](const auto& in_triangles, unsigned int in_first, unsigned int in_count, const Ray& in_ray, HitData& intersection_data)
		{
			bool hit_triangle = false;

			for (unsigned int idx = in_first; idx < in_count; ++idx)
			{
				const uint32_t mesh_id = in_triangles[idx].GetMeshId();
				const uint32_t triangle_id = in_triangles[idx].GetTriangleId() * 3;

				const vec3 v0 = in_triangles[idx].vertices[0];
				const vec3 v1 = in_triangles[idx].vertices[1];
				const vec3 v2 = in_triangles[idx].vertices[2];

				collision::TriangleHitData tri_hit_data(intersection_data.t);
				if (collision::RayTriangle(in_ray, v0, v1, v2, tri_hit_data))
				{
					intersection_data.t = tri_hit_data.RayT;
					intersection_data.uv = tri_hit_data.TriangleUV;
					intersection_data.triangle_index = triangle_id;
					intersection_data.object_index = mesh_id;
					hit_triangle = true;
				}
			}

			return hit_triangle;
		};
		

		if (accel::IntersectsWithTree<Tri>(SceneTree.GetChild(0), ray, intersection_data, TriangleRayTester))
		{
			hit_any_mesh = true;
		}

#else

		for (unsigned int i = 0; i < scene.GetObjectCount(); ++i)
		{
			const Mesh& mesh = scene.GetObject(i);

			if (collision::RayAABB(ray, mesh.GetAABB(), intersection_data.t))
			{
				collision::MeshHitData mesh_hit(intersection_data.t);
				if (collision::RayMesh(ray, mesh, mesh_hit))
				{
					intersection_data.t = mesh_hit.RayT;
					intersection_data.uv = mesh_hit.TriangleUV;
					intersection_data.triangle_index = mesh_hit.TriangleIndex;
					intersection_data.object_index = i;
					hit_any_mesh = true;
				}
			}
		}

#endif

		if (hit_any_mesh)
		{
			FillTriangleIntersectionData(scene.GetObject(intersection_data.object_index), ray, intersection_data);
		}

		return hit_any_mesh;
	}

	void BuildInternalScene(const Scene& scene)
	{
#if USE_KDTREE

		// Triangle-AABB intersection
		auto TriangleAABBTester = [&scene](const auto& in_triangle, const BBox& in_aabb)
		{
			const Mesh& mesh = scene.GetObject(in_triangle.GetMeshId());

			vec3 v0{ in_triangle.vertices[0] - in_aabb.GetCenter() };
			vec3 v1{ in_triangle.vertices[1] - in_aabb.GetCenter() };
			vec3 v2{ in_triangle.vertices[2] - in_aabb.GetCenter() };

			return collision::TriangleAABB(v0, v1, v2, in_aabb);
		};

		accel::Node<Tri> TempTree;
		TempTree.GetElements().reserve(scene.GetTriCount());
		
		BBox scene_bbox{ FLT_MAX, -FLT_MAX };
		for (uint16_t i = 0; i < scene.GetObjectCount(); ++i)
		{
			const Mesh& mesh = scene.GetObject(i);
			for (uint32_t t = 0; t < mesh.GetTriCount(); ++t)
			{
				vec3 v0{ mesh.GetVertex(mesh.GetIndex(t * 3 + 0)).pos };
				vec3 v1{ mesh.GetVertex(mesh.GetIndex(t * 3 + 1)).pos };
				vec3 v2{ mesh.GetVertex(mesh.GetIndex(t * 3 + 2)).pos };

				TempTree.GetElements().emplace_back(i, t, v0, v1, v2);
			}

			scene_bbox.minbound = pmin(mesh.GetAABB().minbound, scene_bbox.minbound);
			scene_bbox.maxbound = pmax(mesh.GetAABB().maxbound, scene_bbox.maxbound);
		}
		TempTree.SetAABB(scene_bbox);
		
		accel::BuildTree<Tri>(&TempTree, TriangleAABBTester);

		accel::FlattenTree<Tri>(TempTree, SceneTree);
	}
	accel::FlatTree<Tri> SceneTree;

#else
	}

#endif

	//
	// -- platform data for rendering --
	//
	vector<vec3> output;
	vector<vec3> traceresult;
	uint32_t* bitmap_bytes;

	int tile_count_ = 0;

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

	details_->tile_count_ = max(win_width_, win_height_) / tile_size_;

	details_->traceresult.resize(in_width * in_height);
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

void CpuTrace::UpdateScene()
{
	static float counter = .0f;
	float blend_factor = counter / (counter + 1);

	// copy last frame result to bitmap for displaying
	#pragma omp parallel for collapse(2)
	for (int j = 0; j < win_height_; ++j)
	{
		for (int i = 0; i < win_width_; ++i)
		{
			int idx = j * win_width_ + i;

			details_->output[idx] = lerp(details_->traceresult[idx], details_->output[idx], blend_factor);
			vec3 bitmap_col = clamp(255.99f * srgb(details_->output[idx]), vec3(.0f), vec3(255.f));
			uint32_t dst = (uint8_t)bitmap_col.b |
				           ((uint8_t)bitmap_col.g << 8) |
				           ((uint8_t)bitmap_col.r << 16);

#if defined(WIN32)
			details_->bitmap_bytes[idx] = dst;
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

	++counter;
}

void CpuTrace::RenderTile(int tile_x, int tile_y, int tile_size, int w, int h)
{
	for (int j = tile_x * tile_size; j < (tile_x + 1) * tile_size; ++j)
	{
		for (int i = tile_y * tile_size; i < (tile_y + 1) * tile_size; ++i)
		{
			int idx = j * w + i;
			if (idx < w * h)
			{
				static uint32_t random_ctx = 0x12345;

				float u = (i + fastrand(random_ctx)) / float(w);
				float v = (j + fastrand(random_ctx)) / float(h);

				details_->traceresult[idx] = Trace(camera_->GetRayFrom(u, v), random_ctx);
			}
		}
	}
}

void CpuTrace::RenderScene()
{
	camera_->BeginFrame();
	
	#pragma omp parallel for collapse(2) schedule(dynamic)
	for (int tile_x = 0; tile_x < details_->tile_count_; ++tile_x)
	{
		for (int tile_y = 0; tile_y < details_->tile_count_; ++tile_y)
		{
			RenderTile(tile_x, tile_y, tile_size_, win_width_, win_height_);
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
