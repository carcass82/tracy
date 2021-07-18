/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#include "cpu_details.h"
#include "scene.h"
#include "collision.h"

bool CPUDetails::Initialize(WindowHandle ctx, uint32_t w, uint32_t h, uint32_t size)
{
	tile_count_ = max(w, h) / size;

	render_data_.width = w;
	render_data_.height = h;
	render_data_.output.resize(static_cast<size_t>(w) * h, vec3{});

	return (render_data_.bitmap.Create(ctx, w, h));
}

void CPUDetails::Shutdown()
{
	render_data_.bitmap.Destroy();
}

bool CPUDetails::ProcessScene(const Scene& scene)
{
#if USE_KDTREE

	// utility wrapper for Object-AABB intersection
	auto ObjectAABBTester = [&scene](const auto& in_object, const BBox& in_aabb)
	{
		const auto& mesh = scene.GetObject(in_object.object_id);
		return in_aabb.Contains(mesh.GetAABB().GetCenter());
	};

	// utility wrapper for Triangle-AABB intersection
	auto TriangleAABBTester = [&scene](const auto& in_triangle, const BBox& in_aabb)
	{
		const auto& mesh = scene.GetObject(in_triangle.GetMeshId());
		
		vec3 v0{ mesh.GetVertex(mesh.GetIndex(in_triangle.GetTriangleId() * 3 + 0)).pos - in_aabb.GetCenter() };
		vec3 v1{ mesh.GetVertex(mesh.GetIndex(in_triangle.GetTriangleId() * 3 + 1)).pos - in_aabb.GetCenter() };
		vec3 v2{ mesh.GetVertex(mesh.GetIndex(in_triangle.GetTriangleId() * 3 + 2)).pos - in_aabb.GetCenter() };

		return collision::TriangleAABB(v0, v1, v2, in_aabb);
	};

	BLAS_tree_.resize(scene.GetObjectCount());
	
	accel::Node<Obj> TempObjectsTree;
	TempObjectsTree.GetElements().reserve(scene.GetObjectCount());

	BBox scene_bbox;
	for (uint16_t i = 0; i < scene.GetObjectCount(); ++i)
	{
		const Mesh& mesh = scene.GetObject(i);

		accel::Node<Tri> TempTrianglesTree;
		TempTrianglesTree.GetElements().reserve(mesh.GetTriCount());

		for (uint32_t t = 0; t < mesh.GetTriCount(); ++t)
		{
			vec3 v0{ mesh.GetVertex(mesh.GetIndex(t * 3 + 0)).pos };
			vec3 v1{ mesh.GetVertex(mesh.GetIndex(t * 3 + 1)).pos };
			vec3 v2{ mesh.GetVertex(mesh.GetIndex(t * 3 + 2)).pos };
			
			TempTrianglesTree.GetElements().emplace_back(i, t, v0, v1, v2);
		}

		TempTrianglesTree.SetAABB(mesh.GetAABB());
		accel::BuildTree<Tri>(&TempTrianglesTree, TriangleAABBTester);
		accel::FlattenTree<Tri>(TempTrianglesTree, BLAS_tree_[i]);

		TempObjectsTree.GetElements().emplace_back(i);
		scene_bbox.Extend(mesh.GetAABB());
	}

	TempObjectsTree.SetAABB(scene_bbox);
	accel::BuildTree<Obj>(&TempObjectsTree, ObjectAABBTester);
	accel::FlattenTree<Obj>(TempObjectsTree, TLAS_tree_);

#endif

	return true;
}

bool CPUDetails::ComputeIntersection(const Scene& scene, const Ray& ray, HitData& data) const
{
	bool hit_any_mesh = false;

#if USE_KDTREE

	const auto TriangleRayTester = [](const Tri* in_triangles, uint32_t in_first, uint32_t in_count, const Ray& in_ray, HitData& intersection_data)
	{
		bool hit_triangle{};

#if USE_INTRINSICS
		vec3 origin = in_ray.GetOrigin();
		vec3 direction = in_ray.GetDirection();

		__m128 rayO{ _mm_set_ps(origin.z, origin.z, origin.y, origin.x) };
		__m128 rayD{ _mm_set_ps(direction.z, direction.z, direction.y, direction.x) };
#else
		const vec3 rayO{ in_ray.GetOrigin() };
		const vec3 rayD{ in_ray.GetDirection() };
#endif

		collision::TriangleHitData tri_hit_data(intersection_data.t);

		for (uint32_t idx = in_first; idx < in_count; ++idx)
		{
			const auto& triangle = in_triangles[idx];

			if (collision::RayTriangle(rayO, rayD, triangle.vertices, tri_hit_data))
			{
				intersection_data.t = tri_hit_data.RayT;
				intersection_data.uv = tri_hit_data.TriangleUV;
				intersection_data.object_index = triangle.GetMeshId();
				intersection_data.triangle_index = triangle.GetTriangleId() * 3;
				hit_triangle = true;
			}
		}

		return hit_triangle;
	};

	const auto ObjectRayTester = [this, TriangleRayTester](const Obj* in_objects, uint32_t in_first, uint32_t in_count, const Ray& in_ray, HitData& intersection_data)
	{
		bool hit_object{ false };

		for (uint32_t idx = in_first; idx < in_count; ++idx)
		{
			uint32_t tree_id = in_objects[idx].object_id;
			if (accel::IntersectsWithTree<Tri>(BLAS_tree_[tree_id].GetChild(0), in_ray, intersection_data, TriangleRayTester))
			{
				hit_object = true;
			}
		}

		return hit_object;
	};

	hit_any_mesh = accel::IntersectsWithTree<Obj>(TLAS_tree_.GetChild(0), ray, data, ObjectRayTester);


#else

	for (uint32_t i = 0; i < scene.GetObjectCount(); ++i)
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

		data.point = ray.GetPoint(data.t);
		data.normal = (1.f - uv.x - uv.y) * v0.normal + uv.x * v1.normal + uv.y * v2.normal;
		data.tangent = (1.f - uv.x - uv.y) * v0.tangent + uv.x * v1.tangent + uv.y * v2.tangent;
		data.uv = (1.f - uv.x - uv.y) * v0.uv0 + uv.x * v1.uv0 + uv.y * v2.uv0;
		data.material = mesh.GetMaterial();
	}

	return hit_any_mesh;
}

void CPUDetails::UpdateOutput(uint32_t index, const vec3& color)
{
#if ACCUMULATE_SAMPLES

	float blend_factor = frame_counter_ / (frame_counter_ + 1.f);
	render_data_.output[index] = lerp(color, render_data_.output[index], blend_factor);

#else

	render_data_.output[index] = color;

#endif
}

void CPUDetails::UpdateBitmap()
{
	// copy last frame result to bitmap for displaying
	#pragma omp parallel for collapse(2)
	for (int32_t j = 0; j < static_cast<int32_t>(render_data_.height); ++j)
	{
		for (int32_t i = 0; i < static_cast<int32_t>(render_data_.width); ++i)
		{
			int32_t idx = j * render_data_.width + i;
			render_data_.bitmap.SetPixel(i, j, Tonemap(render_data_.output[idx]));
		}
	}

	// buffer ready, frame complete
	++frame_counter_;
}

vec3 CPUDetails::Tonemap(const vec3& color) const
{
	static constexpr float kExposure{ TRACY_EXPOSURE };

#if USE_TONEMAP_REINHARD

	using cc::gfx::reinhard;
	vec3 output{ srgb(reinhard(color * kExposure)) };

#elif USE_TONEMAP_ACES
	
	using cc::gfx::aces;
	vec3 output{ srgb(aces(color * kExposure)) };

#elif USE_TONEMAP_SRGB

	vec3 output{ srgb(color * kExposure) };

#else

	vec3 output{ color };

#endif
	
	return clamp(255.99f * output, vec3(.0f), vec3(255.f));
}

void CPUDetails::Render(WindowHandle ctx, uint32_t w, uint32_t h)
{
	render_data_.bitmap.Paint(ctx);
}
