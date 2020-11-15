#include "cpu_details.h"
#include "scene.h"
#include "collision.h"

bool CPUDetails::Initialize(WindowHandle ctx, uint32_t w, uint32_t h, uint32_t size)
{
	tile_count_ = max(w, h) / size;

	render_data_.width = w;
	render_data_.height = h;
	render_data_.output.resize(static_cast<size_t>(w) * h, {});

#if defined(WIN32)
	BITMAPINFO bmi;
	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth = w;
	bmi.bmiHeader.biHeight = h;
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 32;
	bmi.bmiHeader.biCompression = BI_RGB;
	bmi.bmiHeader.biSizeImage = w * h * bmi.bmiHeader.biBitCount / 8;
	HDC hdc = CreateCompatibleDC(GetDC(ctx->win));
	render_data_.bitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&render_data_.bitmap_bytes, nullptr, 0);
#else
	//details_->bitmap_bytes = new uint32_t[in_width * in_height];
	//details_->bitmap = XCreateImage(win_handle_->dpy,
	//                              DefaultVisual(win_handle_->dpy, win_handle_->ds),
	//                              DefaultDepth(win_handle_->dpy, win_handle_->ds),
	//                              ZPixmap,
	//                              0,
	//                              reinterpret_cast<char*>(details_->bitmap_bytes),
	//                              in_width,
	//                              in_height,
	//                              32,
	//                              0);
#endif

	return (render_data_.bitmap != nullptr);
}

void CPUDetails::Shutdown()
{
	DeleteObject(render_data_.bitmap);
}

bool CPUDetails::ProcessScene(const Scene& scene)
{
#if USE_KDTREE

	// Triangle-AABB intersection
	auto TriangleAABBTester = [](const auto& in_triangle, const BBox& in_aabb)
	{
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
	accel::FlattenTree<Tri>(TempTree, scene_tree_);

#endif

	return true;
}

namespace
{
	bool TriangleRayTester(const Tri* in_triangles, unsigned int in_first, unsigned int in_count, const Ray& in_ray, HitData& intersection_data)
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
	}
}

bool CPUDetails::ComputeIntersection(const Scene& scene, const Ray& ray, HitData& data) const
{
	bool hit_any_mesh = false;

#if USE_KDTREE

	if (accel::IntersectsWithTree<Tri>(scene_tree_.GetChild(0), ray, data, TriangleRayTester))
	{
		hit_any_mesh = true;
	}

#else

	for (unsigned int i = 0; i < scene.GetObjectCount(); ++i)
	{
		const Mesh& mesh = scene.GetObject(i);

		if (collision::RayAABB(ray, mesh.GetAABB(), data.t))
		{
			collision::MeshHitData mesh_hit(data.t);
			if (collision::RayMesh(ray, mesh, mesh_hit))
			{
				data.t = mesh_hit.RayT;
				data.uv = mesh_hit.TriangleUV;
				data.triangle_index = mesh_hit.TriangleIndex;
				data.object_index = i;
				hit_any_mesh = true;
			}
		}
	}

#endif

	if (hit_any_mesh)
	{
		const Mesh& mesh = scene.GetObject(data.object_index);
		const Vertex v0 = mesh.GetVertex(mesh.GetIndex(data.triangle_index + 0));
		const Vertex v1 = mesh.GetVertex(mesh.GetIndex(data.triangle_index + 1));
		const Vertex v2 = mesh.GetVertex(mesh.GetIndex(data.triangle_index + 2));
		const vec2 uv = data.uv;
		const Material* material = mesh.GetMaterial();

		data.point = ray.GetPoint(data.t);
		data.normal = (1.f - uv.x - uv.y) * v0.normal + uv.x * v1.normal + uv.y * v2.normal;
		data.uv = (1.f - uv.x - uv.y) * v0.uv0 + uv.x * v1.uv0 + uv.y * v2.uv0;
		data.material = material;
	}

	return hit_any_mesh;
}

void CPUDetails::UpdateOutput(uint32_t index, const vec3& color)
{
	float blend_factor = frame_counter_ / (frame_counter_ + 1.f);
	render_data_.output[index] = lerp(color, render_data_.output[index], blend_factor);
}

void CPUDetails::UpdateBitmap()
{
	// copy last frame result to bitmap for displaying
	#pragma omp parallel for collapse(2)
	for (int32_t j = 0; j < static_cast<int32>(render_data_.height); ++j)
	{
		for (int32_t i = 0; i < static_cast<int32>(render_data_.width); ++i)
		{
			int32_t idx = j * render_data_.width + i;

			vec3 bitmap_col = Tonemap(render_data_.output[idx]);

			uint32_t dst =  (uint8_t)bitmap_col.b       |
			               ((uint8_t)bitmap_col.g << 8) |
			               ((uint8_t)bitmap_col.r << 16);

#if defined(WIN32)
			render_data_.bitmap_bytes[idx] = dst;
		}
	}

#else
	#error "TODO: review!"
	//		XPutPixel(details_->bitmap, i, win_height_ - j, dst);
	//	}
	//}
	//XPutImage(win_handle_->dpy, win_handle_->win, DefaultGC(win_handle_->dpy, win_handle_->ds), details_->bitmap, 0, 0, 0, 0, win_width_, win_height_);
	//XFlush(win_handle_->dpy);
#endif

	// buffer ready, frame complete
	++frame_counter_;
}

vec3 CPUDetails::Tonemap(const vec3& color)
{
	// perhaps add a better "tonemapping" than Linear -> sRGB
	return clamp(255.99f * srgb(color), vec3(.0f), vec3(255.f));
}

void CPUDetails::Render(WindowHandle ctx, uint32_t w, uint32_t h)
{
#if defined(WIN32)
	PAINTSTRUCT ps;
	RECT rect;
	HDC hdc = BeginPaint(ctx->win, &ps);
	GetClientRect(ctx->win, &rect);

	HDC srcDC = CreateCompatibleDC(hdc);
	SetStretchBltMode(hdc, COLORONCOLOR);
	SelectObject(srcDC, render_data_.bitmap);
	StretchBlt(hdc, 0, 0, rect.right, rect.bottom, srcDC, 0, 0, w, h, SRCCOPY);
	DeleteObject(srcDC);

	EndPaint(ctx->win, &ps);
#endif
}
