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

#if defined(_OPENMP)
 #include <omp.h>
#endif

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
	const int32_t w = in_Scene.Width();
	const int32_t h = in_Scene.Height();

	#pragma omp parallel
	{
		static RandomCtxData random_ctx{ 0x12345 };

		#pragma omp for collapse(2) schedule(dynamic)

#if TILED_RENDERING
		for (int32_t tile_x = 0; tile_x < static_cast<int32_t>(Details.GetTileCount()); ++tile_x)
		{
			for (int32_t tile_y = 0; tile_y < static_cast<int32_t>(Details.GetTileCount()); ++tile_y)
			{
				RenderTile(tile_x, tile_y, kTileSize, in_Scene, random_ctx);
			}
		}
#else
		for (int32_t x = 0; x < w; ++x)
		{
			for (int32_t y = 0; y < h; ++y)
			{
				int32_t idx{ y * w + x };
				float u{ (x + fastrand(random_ctx)) / float(w) };
				float v{ (y + fastrand(random_ctx)) / float(h) };

				Details.UpdateOutput(idx, Trace(in_Scene.GetCamera().GetRayFrom(u, v), in_Scene, random_ctx));
			}
		}
#endif
	}

	Details.UpdateBitmap();
}

#if TILED_RENDERING
void CpuTrace::RenderTile(uint32_t in_TileX, uint32_t in_TileY, uint32_t in_TileSize, const Scene& in_Scene, RandomCtx random_ctx)
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
				float u = (i + fastrand(random_ctx)) / float(w);
				float v = (j + fastrand(random_ctx)) / float(h);

				Details.UpdateOutput(idx, Trace(in_Scene.GetCamera().GetRayFrom(u, v), in_Scene, random_ctx));
			}
		}
	}
}
#endif

vec3 CpuTrace::Trace(const Ray& ray, const Scene& scene, RandomCtx random_ctx)
{
	Ray current_ray{ ray.GetOrigin(), ray.GetDirection() };
	vec3 current_color{ 1.f, 1.f, 1.f };

	for (int t = 0; t < kBounces; ++t)
	{
		++raycount_;

		collision::HitData intersection_data;
		intersection_data.t = FLT_MAX;

		if (Details.ComputeIntersection(scene, current_ray, intersection_data))
		{
#if DEBUG_SHOW_NORMALS
			return .5f * normalize((1.f + mat3(scene.GetCamera().GetView()) * intersection_data.normal));
#else
			vec3 attenuation;
			vec3 emission;
			if (intersection_data.material->Scatter(current_ray, intersection_data, attenuation, emission, current_ray, random_ctx))
			{
				current_color *= attenuation;
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
