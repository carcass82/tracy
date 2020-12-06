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

void CpuTrace::OnUpdate(const Scene& in_Scene, float in_DeltaTime)
{
#if !TILED_RENDERING
	const int32_t w{ static_cast<int32_t>(in_Scene.Width()) };
	const int32_t h{ static_cast<int32_t>(in_Scene.Height()) };
#endif

	#pragma omp parallel
	{
		static RandomCtxData random_ctx{ initrand() };

		#pragma omp for collapse(2) schedule(dynamic)

#if TILED_RENDERING
		for (int32_t tile_x = 0; tile_x < static_cast<int32_t>(Details.GetTileCount()); ++tile_x)
		{
			for (int32_t tile_y = 0; tile_y < static_cast<int32_t>(Details.GetTileCount()); ++tile_y)
			{
				RenderTile(tile_x, tile_y, in_Scene, random_ctx);
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

void CpuTrace::OnEvent(TracyEvent in_Event, const WindowHandle in_Window, const Scene& in_Scene)
{
	switch (in_Event)
	{
	case TracyEvent::eCameraCut:
		Details.ResetFrameCounter();
		break;
	default:
		break;
	}
}

#if TILED_RENDERING
void CpuTrace::RenderTile(uint32_t tile_x, uint32_t tile_y, const Scene& scene, RandomCtx random_ctx)
{
	uint32_t w = scene.Width();
	uint32_t h = scene.Height();

	for (uint32_t j = tile_x * kTileSize; j < (tile_x + 1) * kTileSize; ++j)
	{
		for (uint32_t i = tile_y * kTileSize; i < (tile_y + 1) * kTileSize; ++i)
		{
			uint32_t idx = j * w + i;
			if (idx < w * h)
			{
				float u = (i + fastrand(random_ctx)) / float(w);
				float v = (j + fastrand(random_ctx)) / float(h);

				Details.UpdateOutput(idx, Trace(scene.GetCamera().GetRayFrom(u, v), scene, random_ctx));
			}
		}
	}
}
#endif

vec3 CpuTrace::Trace(Ray&& ray, const Scene& scene, RandomCtx random_ctx)
{
	Ray current_ray{ std::move(ray) };
	vec3 throughput{ 1.f, 1.f, 1.f };
	vec3 pixel;

	for (uint32_t t = 0; t < kBounces; ++t)
	{
		++raycount_;

		collision::HitData intersection_data;
		intersection_data.t = FLT_MAX;

		vec3 attenuation;
		vec3 emission;

		if (Details.ComputeIntersection(scene, current_ray, intersection_data))
		{

#if DEBUG_SHOW_BASECOLOR
			return intersection_data.material->GetBaseColor(intersection_data);
#elif DEBUG_SHOW_NORMALS
			return .5f * normalize((1.f + mat3(scene.GetCamera().GetView()) * intersection_data.material->GetNormal(intersection_data)));
#elif DEBUG_SHOW_METALNESS
			return vec3(intersection_data.material->GetMetalness(intersection_data));
#elif DEBUG_SHOW_ROUGHNESS
			return vec3(intersection_data.material->GetRoughness(intersection_data));
#elif DEBUG_SHOW_EMISSIVE
			return intersection_data.material->GetEmissive(intersection_data);
#endif

			intersection_data.material->Scatter(current_ray, intersection_data, attenuation, emission, current_ray, random_ctx);
			{
				pixel += emission * throughput;
				throughput *= attenuation;
			}
		}
		else
		{
			vec3 v{ normalize(current_ray.GetDirection()) };
			intersection_data.uv = vec2(atan2f(v.z, v.x) / (2 * PI), asinf(v.y) / PI) + 0.5f;
			scene.GetSkyMaterial()->Scatter(current_ray, intersection_data, attenuation, emission, current_ray, random_ctx);
			
			pixel += emission * throughput;
			break;
		}

#if USE_RUSSIAN_ROULETTE
		float p = max(throughput.r, max(throughput.g, throughput.b));
		if (fastrand(random_ctx) > p)
		{
			break;
		}
		
		throughput *= rcp(p);
#endif
	}

	return pixel;
}

void CpuTrace::OnRender(const WindowHandle in_Window)
{
	if (LIKELY(IsValidWindowHandle(in_Window)))
	{
		Details.Render(in_Window, in_Window->width, in_Window->height);
	}
}
