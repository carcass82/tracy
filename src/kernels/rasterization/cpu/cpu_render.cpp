/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#include "cpu_render.h"
#include "log.h"
#include "scene.h"
#include "bitmap.h"

namespace
{
	Bitmap bitmap;
	vector<float> depth_buffer;

	constexpr inline vec4 Raster(const vec4& v, uint32_t w, uint32_t h)
	{
		return vec4{ w * (v.x + v.w) * .5f, h * (v.w - v.y) * .5f, v.z, v.w };
	}

	constexpr inline bool EdgeEval(const vec3& edge, const vec3& sample)
	{
		// Interpolate edge function at given sample
		float result = (edge.x * sample.x) + (edge.y * sample.y) + edge.z;

		// Apply tie-breaking rules on shared vertices in order to avoid double-shading fragments
		if (result > .0f) return true;
		else if (result < .0f) return false;

		if (edge.x > .0f) return true;
		else if (edge.x < .0f) return false;

		if (edge.y < .0f) return false;

		return true;
	}

	struct VSInput
	{
		mat4 Model;
		mat4 View;
		mat4 Projection;
		mat4 ModelViewProjection;
	};

	struct FSInput
	{
		vec3 normal;
		vec2 uv0;
	};

	constexpr inline vec4 VS(const Vertex& v, const VSInput& vs_in, FSInput& vs_out)
	{
		vs_out.normal = v.normal;
		vs_out.uv0 = v.uv0;

		return vs_in.ModelViewProjection * vec4(v.pos, 1.f);
	}

	constexpr inline vec4 FS(const FSInput& fs_data)
	{
		return vec4(fs_data.normal * vec3(0.5) + vec3(0.5), 1.f);
	}
}


bool CPURaster::Startup(const WindowHandle in_Window, const Scene& in_Scene)
{
	depth_buffer.resize(in_Window->width * in_Window->height, FLT_MAX);

	return bitmap.Create(in_Window, in_Window->width, in_Window->height);
}

void CPURaster::OnUpdate(const Scene& in_Scene, float in_DeltaTime)
{
	const uint32_t w = in_Scene.GetWidth();
	const uint32_t h = in_Scene.GetHeight();

	VSInput vs_data;
	vs_data.Model = mat4(1.f); // TODO: update when adding object instancing
	vs_data.View = in_Scene.GetCamera().GetView();
	vs_data.Projection = in_Scene.GetCamera().GetProjection();
	vs_data.ModelViewProjection = vs_data.Projection * vs_data.View * vs_data.Model;

	for (const auto& object : in_Scene.GetObjects())
	{
		for (uint32_t i = 0; i < object.GetTriCount(); ++i)
		{
			const Vertex vertices[]
			{
				object.GetVertex(object.GetIndex(i * 3 + 0)),
				object.GetVertex(object.GetIndex(i * 3 + 1)),
				object.GetVertex(object.GetIndex(i * 3 + 2))
			};

			FSInput vs_out[3];

			const vec4 transformed[]
			{
				VS(vertices[0], vs_data, vs_out[0]),
				VS(vertices[1], vs_data, vs_out[1]),
				VS(vertices[2], vs_data, vs_out[2])
			};

			const vec4 homogeneous[]
			{
				Raster(transformed[0], w, h),
				Raster(transformed[1], w, h),
				Raster(transformed[2], w, h)
			};

			mat3 vertex_matrix
			{
				vec3{ homogeneous[0].x, homogeneous[1].x, homogeneous[2].x },
				vec3{ homogeneous[0].y, homogeneous[1].y, homogeneous[2].y },
				vec3{ homogeneous[0].w, homogeneous[1].w, homogeneous[2].w }
			};

			// If det(M) == 0.0f, we'd perform division by 0 when calculating the invert matrix,
			// whereas (det(M) > 0) implies a back-facing triangle
			if (determinant(vertex_matrix) < 0.0f)
			{
				// Compute the inverse of vertex matrix to use it for setting up edge functions
				vertex_matrix = inverse(vertex_matrix);

				// Calculate edge functions based on the vertex matrix
				vec3 E0 = vertex_matrix[0] / (abs(vertex_matrix[0].x) + abs(vertex_matrix[0].y));
				vec3 E1 = vertex_matrix[1] / (abs(vertex_matrix[1].x) + abs(vertex_matrix[1].y));
				vec3 E2 = vertex_matrix[2] / (abs(vertex_matrix[2].x) + abs(vertex_matrix[2].y));

				// Calculate constant function to interpolate 1/w
				vec3 C = vertex_matrix * vec3(1.f);

				// Calculate z interpolation vector
				vec3 Z = vertex_matrix * vec3(transformed[0].z, transformed[1].z, transformed[2].z);

				// Calculate normal interpolation vector
				vec3 PNX = vertex_matrix * vec3(vs_out[0].normal.x, vs_out[1].normal.x, vs_out[2].normal.x);
				vec3 PNY = vertex_matrix * vec3(vs_out[0].normal.y, vs_out[1].normal.y, vs_out[2].normal.y);
				vec3 PNZ = vertex_matrix * vec3(vs_out[0].normal.z, vs_out[1].normal.z, vs_out[2].normal.z);

				// Calculate UV interpolation vector
				vec3 PUVS = vertex_matrix * vec3(vs_out[0].uv0.s, vs_out[1].uv0.s, vs_out[2].uv0.s);
				vec3 PUVT = vertex_matrix * vec3(vs_out[0].uv0.t, vs_out[1].uv0.t, vs_out[2].uv0.t);

				for (uint32_t x = 0; x < w; ++x)
				{
					for (uint32_t y = 0; y < h; ++y)
					{
						const uint32_t pixel_idx{ y * w + x };

						// Sample location at the center of each pixel
						vec3 sample{ x + .5f, y + .5f, 1.f };

						// Evaluate edge functions at every fragment
						// If sample is "inside" of all three half-spaces bounded by the three edges of our triangle, it's 'on' the triangle
						if (EdgeEval(E0, sample) && EdgeEval(E1, sample) && EdgeEval(E2, sample))
						{
							// Interpolate 1/w at current fragment
							float oneOverW = (C.x * sample.x) + (C.y * sample.y) + C.z;
							float w = 1.f / oneOverW;

							// Interpolate z that will be used for depth test
							float zOverW = (Z.x * sample.x) + (Z.y * sample.y) + Z.z;
							float z = zOverW * w;

							// Perform depth test with interpolated z value
							if (z <= depth_buffer[pixel_idx])
							{
								// Depth test passed; update depth buffer value
								depth_buffer[pixel_idx] = z;

								// Interpolate normal
								float nxOverW = (PNX.x * sample.x) + (PNX.y * sample.y) + PNX.z;
								float nyOverW = (PNY.x * sample.x) + (PNY.y * sample.y) + PNY.z;
								float nzOverW = (PNZ.x * sample.x) + (PNZ.y * sample.y) + PNZ.z;

								// Interpolate texture coordinates
								float uOverW = (PUVS.x * sample.x) + (PUVS.y * sample.y) + PUVS.z;
								float vOverW = (PUVT.x * sample.x) + (PUVT.y * sample.y) + PUVT.z;

								// Final vertex attributes to be passed to FS
								FSInput fragment_data = { vec3(nxOverW, nyOverW, nzOverW) * w, vec2(uOverW, vOverW) * w };

								// TODO: alpha blending
								vec4 color = clamp(255.99f * FS(fragment_data), vec4(.0f), vec4(255.f));
								bitmap.SetPixel(x, h - y, color.rgb);
							}
						}
					}
				}
			}
		}
	}
}

void CPURaster::OnRender(const WindowHandle in_Window)
{
	if LIKELY(IsValidWindowHandle(in_Window))
	{
		bitmap.Paint(in_Window);
	}
}

void CPURaster::Shutdown()
{
}
