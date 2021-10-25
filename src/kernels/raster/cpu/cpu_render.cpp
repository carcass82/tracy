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

	constexpr inline vec4 Raster(const vec4& v, u32 w, u32 h)
	{
		return vec4{ w * (v.x + v.w) * .5f, h * (v.w - v.y) * .5f, v.z, v.w };
	}

	constexpr inline bool TriangleEval(const vec3 edges[3], const vec3& sample)
	{
		// If sample is "inside" of all three half-spaces bounded by the three edges of our triangle, it's 'on' the triangle
		for (i32 i = 0; i < 3; ++i)
		{
			vec3 edge = edges[i];

			// Interpolate edge function at given sample
			float result = (edge.x * sample.x) + (edge.y * sample.y) + edge.z;

			// Apply tie-breaking rules on shared vertices in order to avoid double-shading fragments
			if (result > .0f) continue;
			else if (result < .0f) return false;

			if (edge.x > .0f) continue;
			else if (edge.x < .0f) return false;

			if (edge.y < .0f) return false;
		}
		
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
		vec3 tangent;
		vec3 bitangent;
		vec2 uv0;
	};

	struct FSInterpolator
	{
		mat3 normal;
		mat3 tangent;
		mat3 bitangent;
		mat3 uv0;
	};

	constexpr inline vec4 VS(const Vertex& v, const VSInput& vs_in, FSInput& vs_out)
	{
		vs_out.normal = v.normal;
		vs_out.tangent = v.tangent;
		vs_out.bitangent = v.bitangent;
		vs_out.uv0 = { v.uv0.s, 1.f - v.uv0.t };

		return vs_in.ModelViewProjection * vec4(v.pos, 1.f);
	}

	constexpr inline vec4 FS(const Material& material, const FSInput& fs_data)
	{
		mat3 TBN = mat3(normalize(fs_data.bitangent), normalize(fs_data.tangent), normalize(fs_data.normal));

#if DEBUG_SHOW_BASECOLOR
		return vec4(material.GetAlbedo(), 1.f);
#elif DEBUG_SHOW_NORMALS
		return vec4(material.GetNormal() * .5 + .5, 1.f);
#elif DEBUG_SHOW_METALNESS
		return vec4(material.GetMetalness(), 1.f);
#elif DEBUG_SHOW_ROUGHNESS
		return vec4(material.GetRoughness(), 1.f);
#elif DEBUG_SHOW_EMISSIVE
		return vec4(material.GetEmissive(), 1.f);
#else
		return vec4(material.GetAlbedo(), 1.f);
#endif
	}
}


bool CPURender::Startup(const WindowHandle in_Window, const Scene& in_Scene)
{
	depth_buffer.resize(in_Window->width * in_Window->height, FLT_MAX);

	return bitmap.Create(in_Window, in_Window->width, in_Window->height);
}

void CPURender::OnUpdate(const Scene& in_Scene, float in_DeltaTime)
{
	const i32 w = static_cast<i32>(in_Scene.GetWidth());
	const i32 h = static_cast<i32>(in_Scene.GetHeight());

	VSInput vs_data;
	vs_data.Model = mat4(1.f); // TODO: update when adding object instancing
	vs_data.View = in_Scene.GetCamera().GetView();
	vs_data.Projection = in_Scene.GetCamera().GetProjection();
	vs_data.ModelViewProjection = vs_data.Projection * vs_data.View * vs_data.Model;

	// clear depth
	bitmap.Clear(vec3(.0f));
	fill(depth_buffer.begin(), depth_buffer.end(), FLT_MAX);

	for (const auto& object : in_Scene.GetObjects())
	{
		const Material& object_material = in_Scene.GetMaterial(object.GetMaterial());

		for (u32 i = 0; i < object.GetTriCount(); ++i)
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
			if (determinant(vertex_matrix) < .0f)
			{
				// Compute the inverse of vertex matrix to use it for setting up edge functions
				vertex_matrix = inverse(vertex_matrix);

				// Calculate edge functions based on the vertex matrix
				vec3 edges[] =
				{
					vertex_matrix[0] / (abs(vertex_matrix[0].x) + abs(vertex_matrix[0].y)),
					vertex_matrix[1] / (abs(vertex_matrix[1].x) + abs(vertex_matrix[1].y)),
					vertex_matrix[2] / (abs(vertex_matrix[2].x) + abs(vertex_matrix[2].y))
				};

				// Calculate constant function to interpolate 1/w
				vec3 C = vertex_matrix * vec3(1.f);

				// Calculate z interpolation vector
				vec3 Z = vertex_matrix * vec3(transformed[0].z, transformed[1].z, transformed[2].z);

				FSInterpolator interpolator_helper;
				interpolator_helper.normal = mat3
				{
					vertex_matrix * vec3(vs_out[0].normal.x, vs_out[1].normal.x, vs_out[2].normal.x),
					vertex_matrix * vec3(vs_out[0].normal.y, vs_out[1].normal.y, vs_out[2].normal.y),
					vertex_matrix * vec3(vs_out[0].normal.z, vs_out[1].normal.z, vs_out[2].normal.z)
				};

				interpolator_helper.tangent = mat3 
				{
					vertex_matrix * vec3(vs_out[0].tangent.x, vs_out[1].tangent.x, vs_out[2].tangent.x),
					vertex_matrix * vec3(vs_out[0].tangent.y, vs_out[1].tangent.y, vs_out[2].tangent.y),
					vertex_matrix * vec3(vs_out[0].tangent.z, vs_out[1].tangent.z, vs_out[2].tangent.z)
				};

				interpolator_helper.bitangent = mat3
				{
					vertex_matrix * vec3(vs_out[0].bitangent.x, vs_out[1].bitangent.x, vs_out[2].bitangent.x),
					vertex_matrix * vec3(vs_out[0].bitangent.y, vs_out[1].bitangent.y, vs_out[2].bitangent.y),
					vertex_matrix * vec3(vs_out[0].bitangent.z, vs_out[1].bitangent.z, vs_out[2].bitangent.z)
				};

				interpolator_helper.uv0 = mat3
				{
					vertex_matrix * vec3(vs_out[0].uv0.s, vs_out[1].uv0.s, vs_out[2].uv0.s),
					vertex_matrix * vec3(vs_out[0].uv0.t, vs_out[1].uv0.t, vs_out[2].uv0.t),
					vec3(1.f)
				};

				#pragma omp parallel for collapse(2)
				for (i32 x = 0; x < w; ++x)
				{
					for (i32 y = 0; y < h; ++y)
					{
						const i32 pixel_idx{ y * w + x };

						// Sample location at the center of each pixel
						vec3 sample{ x + .5f, y + .5f, 1.f };

						// Evaluate edge functions at every fragment
						if (TriangleEval(edges, sample))
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

								// Final vertex attributes to be passed to FS
								FSInput fragment_data;
								fragment_data.normal = (sample * interpolator_helper.normal) * w;
								fragment_data.tangent = (sample * interpolator_helper.tangent) * w;
								fragment_data.bitangent = (sample * interpolator_helper.bitangent) * w;
								fragment_data.uv0 = ((sample * interpolator_helper.uv0) * w).xy;

								// TODO: additional buffer for alpha blending
								vec4 fragment = FS(object_material, fragment_data);
								vec4 color = clamp(255.99f * fragment, vec4(.0f), vec4(255.f));
								bitmap.SetPixel(x, h - y, color.rgb);
							}
						}
					}
				}
			}
		}
	}
}

void CPURender::OnRender(const WindowHandle in_Window)
{
	if LIKELY(IsValidWindowHandle(in_Window))
	{
		bitmap.Paint(in_Window);
	}
}

void CPURender::Shutdown()
{
}
