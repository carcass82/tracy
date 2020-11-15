/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "cpu_trace.h"
#include "cpu_details.h"
#include "random.h"
#include "collision.h"
#include "scene.h"

namespace
{
	CPUDetails Details;
}

bool CpuTrace::Startup(const WindowHandle in_Window, const Scene& in_Scene)
{
	return Details.Initialize(in_Window, in_Window->width, in_Window->height, kTileSize) &&
	       Details.ProcessScene(in_Scene);
}

void CpuTrace::Shutdown()
{
	Details.Shutdown();
}

void CpuTrace::OnUpdate(const Scene& in_Scene)
{
	#pragma omp parallel for collapse(2) schedule(dynamic)
	for (int32_t tile_x = 0; tile_x < static_cast<int32_t>(Details.GetTileCount()); ++tile_x)
	{
		for (int32_t tile_y = 0; tile_y < static_cast<int32_t>(Details.GetTileCount()); ++tile_y)
		{
			RenderTile(tile_x, tile_y, kTileSize, in_Scene);
		}
	}

	Details.UpdateBitmap();
}

void CpuTrace::RenderTile(uint32_t in_TileX, uint32_t in_TileY, uint32_t in_TileSize, const Scene& in_Scene)
{
	uint32_t w = in_Scene.Width();
	uint32_t h = in_Scene.Height();

	for (uint32_t j = in_TileX * in_TileSize; j < (in_TileX + 1) * in_TileSize; ++j)
	{
		for (uint32_t i = in_TileY * in_TileSize; i < (in_TileY + 1) * in_TileSize; ++i)
		{
			uint32_t idx = j * w + i;
			if (idx < w * h)
			{
				// thread-private but lazy written
				static uint32_t random_ctx = 0x12345 + idx;

				float u = (i + fastrand(random_ctx)) / float(w);
				float v = (j + fastrand(random_ctx)) / float(h);

				Details.UpdateOutput(idx, Trace(in_Scene.GetCamera().GetRayFrom(u, v), in_Scene, random_ctx));
			}
		}
	}
}

vec3 CpuTrace::Trace(const Ray& ray, const Scene& scene, uint32_t random_ctx)
{
	Ray current_ray{ ray };
	vec3 current_color{ 1.f, 1.f, 1.f };

	for (int t = 0; t < kBounces; ++t)
	{
		++raycount_;

		HitData intersection_data;
		intersection_data.t = FLT_MAX;

		if (Details.ComputeIntersection(scene, current_ray, intersection_data))
		{
#if DEBUG_SHOW_NORMALS
			return .5f * normalize((1.f + mat3(scene.GetCamera().GetView()) * intersection_data.normal));
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
			scene.GetSkyMaterial()->Scatter(current_ray, intersection_data, dummy_vec, sky_color, dummy_ray, random_ctx);

			current_color *= sky_color;
			return current_color;
		}
	}

	return {};
}

void CpuTrace::OnRender(const WindowHandle in_Window)
{
	if (LIKELY(IsValidWindowHandle(in_Window)))
	{
		Details.Render(in_Window, in_Window->width, in_Window->height);
	}
}
