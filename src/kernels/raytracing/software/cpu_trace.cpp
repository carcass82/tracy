/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "cpu_trace.h"
#include "random.h"
#include <cfloat>


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
	//
	// https://tavianator.com/fast-branchless-raybounding-box-intersections/
	//
	bool IntersectsWithBoundingBox(const BBox& box, const Ray& ray, float nearest_intersection = FLT_MAX)
	{
		const vec3 inv_ray = ray.GetInvDirection();
		const vec3 minbound = (box.minbound - ray.GetOrigin()) * inv_ray;
		const vec3 maxbound = (box.maxbound - ray.GetOrigin()) * inv_ray;

		vec3 tmin1 = pmin(minbound, maxbound);
		vec3 tmax1 = pmax(minbound, maxbound);

		float tmin = max(tmin1.x, max(tmin1.y, tmin1.z));
		float tmax = min(tmax1.x, min(tmax1.y, tmax1.z));

		return (tmax >= max(1.e-6f, tmin) && tmin < nearest_intersection);
	}

	bool IntersectsWithMesh(const Mesh& mesh, const Ray& in_ray, HitData& inout_intersection)
	{
		bool hit_triangle = false;

		int triangle_idx = -1;
		vec2 triangle_uv{};

		for (int i = 0; i < mesh.GetTriCount(); ++i)
		{
			const Index i0 = mesh.GetIndex(i * 3);
			const Index i1 = mesh.GetIndex(i * 3 + 1);
			const Index i2 = mesh.GetIndex(i * 3 + 2);

			const vec3& v0 = mesh.GetVertex(i0).pos;
			const vec3& v1 = mesh.GetVertex(i1).pos;
			const vec3& v2 = mesh.GetVertex(i2).pos;

			const vec3& v0v1 = v1 - v0;
			const vec3& v0v2 = v2 - v0;

			vec3 pvec = cross(in_ray.GetDirection(), v0v2);
			float det = dot(v0v1, pvec);

			// if the determinant is negative the triangle is backfacing
			// if the determinant is close to 0, the ray misses the triangle
			if (det < 1.e-6f)
			{
				continue;
			}

			float invDet = 1.f / det;

			vec3 tvec = in_ray.GetOrigin() - v0;
			float u = dot(tvec, pvec) * invDet;
			if (u < .0f || u > 1.f)
			{
				continue;
			}

			vec3 qvec = cross(tvec, v0v1);
			float v = dot(in_ray.GetDirection(), qvec) * invDet;
			if (v < .0f || u + v > 1.f)
			{
				continue;
			}

			float t = dot(v0v2, qvec) * invDet;
			if (t < inout_intersection.t && t > 1.e-3f)
			{
				inout_intersection.t = dot(v0v2, qvec) * invDet;
				triangle_uv = vec2{ u, v };
				triangle_idx = i * 3;
				hit_triangle = true;
			}
		}

		if (hit_triangle)
		{
			FillTriangleIntersectionData(mesh, in_ray, triangle_idx, triangle_uv, inout_intersection);
		}

		return hit_triangle;
	}

	void FillTriangleIntersectionData(const Mesh& mesh, const Ray& in_ray, int triangle_idx, const vec2& triangle_uv, HitData& inout_intersection)
	{
		const Vertex& v0 = mesh.GetVertex(mesh.GetIndex(triangle_idx + 0));
		const Vertex& v1 = mesh.GetVertex(mesh.GetIndex(triangle_idx + 1));
		const Vertex& v2 = mesh.GetVertex(mesh.GetIndex(triangle_idx + 2));

		inout_intersection.point = in_ray.GetPoint(inout_intersection.t);
		inout_intersection.normal = (1.f - triangle_uv.x - triangle_uv.y) * v0.normal + triangle_uv.x * v1.normal + triangle_uv.y * v2.normal;
		inout_intersection.uv = (1.f - triangle_uv.x - triangle_uv.y) * v0.uv0 + triangle_uv.x * v1.uv0 + triangle_uv.y * v2.uv0;
		inout_intersection.material = mesh.GetMaterial();
	}

	bool ComputeIntersection(const Scene& scene, const Ray& ray, HitData& intersection_data)
	{
		bool hit_any_mesh = false;
		for (const Mesh& mesh : scene.GetObjects())
		{
			if (IntersectsWithBoundingBox(mesh.GetAABB(), ray, intersection_data.t))
			{
				if (IntersectsWithMesh(mesh, ray, intersection_data))
				{
					hit_any_mesh = true;
				}
			}
		}

		return hit_any_mesh;
	}

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

vec3 CpuTrace::Trace(const Ray& ray)
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
			if (intersection_data.material->Scatter(current_ray, intersection_data, attenuation, emission, scattered))
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
			return {};
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

	#pragma omp parallel for collapse(3) schedule(dynamic)
	for (int j = 0; j < win_height_; ++j)
	{
		for (int i = 0; i < win_width_; ++i)
		{
			for (int s = 0; s < samples_; ++s)
			{
				vec2 uv{ (i + fastrand()) / float(win_width_), (j + fastrand()) / float(win_height_) };

				vec3 sampled_col = Trace(camera_->GetRayFrom(uv.x, uv.y));
				sampled_col /= (float)samples_;

				#pragma omp critical
				{
					vec3 old_col = details_->output[j * win_width_ + i];
					details_->output[j * win_width_ + i] = lerp(sampled_col, old_col, blend_factor);
				}
			}
		}
	}

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
