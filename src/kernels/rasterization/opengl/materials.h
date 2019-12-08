/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

namespace shaders
{
	//
	// common VS
	// 
	const char* vs_source = R"vs(
    #version 330
	layout (location = 0) in vec3 position;
	layout (location = 1) in vec3 normal;
	layout (location = 2) in vec2 uv;
	layout (location = 3) in vec3 tangent;
	layout (location = 4) in vec3 bitangent;

	layout (std140) uniform matrices
	{
		mat4 projection;
		mat4 view;
	};

	out VS_OUT
	{
		vec3 position;
		vec3 normal;
		vec3 light;
		vec2 uv;
	} vs_out;

    void main()
    {
		// fake light position
		vec4 light = vec4(2.0, 5.0, -1.0, 1.0);

		vs_out.position = normalize(view * vec4(position, 1.0)).xyz;
		vs_out.normal = normalize(mat3(view) * normal);
		vs_out.light = normalize(view * light).xyz;
		vs_out.uv = uv;

		gl_Position = projection * view * vec4(position, 1.0);
    })vs";

#if !DEBUG_SHOW_NORMALS
	//
	// standard FS
	// 
	const char* fs_source = R"fs(
	#version 330

	in VS_OUT
	{
		vec3 position;
		vec3 normal;
		vec3 light;
		vec2 uv;
	} vs_in;

	struct Material
	{
		vec3 albedo;
		float metalness;
		float roughness;
		float ior;
	};
	uniform Material material;

	out vec4 out_color;

	void main()
	{
		vec3 n = normalize(vs_in.normal);
		vec3 v = normalize(/* view */ - vs_in.position);
		vec3 l = normalize(vs_in.light - vs_in.position);
		vec3 h = normalize(l + v);
		vec3 r = normalize(reflect(-l, n));

		float NdotL = max(dot(n, l), 0.0);
		float NdotH = max(dot(n, h), 0.0);
		
		

		vec3 direct = (NdotL + pow(NdotH, exp2(10 * material.roughness + 1) )) * material.albedo;
		
		vec3 indirect = vec3(0, 0, 0);

		out_color = vec4(direct + indirect, 1.0);
	}
	)fs";

#else

	//
	// debug FS
	// 
	const char* fs_source = R"fs(
	#version 330

	in VS_OUT
	{
		vec3 position;
		vec3 normal;
		vec3 light;
		vec2 uv;
	} vs_in;

	uniform vec3 albedo = vec3(0.8, 0.1, 0.1);
	uniform float metalness = 0.0;
	uniform float roughness = 0.0;
	uniform float ior = 1.0;

	out vec4 out_color;

	void main()
	{
		vec3 normal_color = (vec3(1.0) + vs_in.normal) * vec3(0.5);
		out_color = vec4(normal_color, 1.0);
	}
	)fs";

#endif
}
