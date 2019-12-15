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
	static const char* vs_source = R"vs(
    #version 330
	layout (location = 0) in vec3 position;
	layout (location = 1) in vec3 normal;
	layout (location = 2) in vec2 uv;
	layout (location = 3) in vec3 tangent;
	layout (location = 4) in vec3 bitangent;

	struct Light
	{
		vec3 position;
		vec3 color;
		float constant;
		float linear;
		float quadratic;
	};
	uniform Light light;

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
		vs_out.position = normalize(view * vec4(position, 1.0)).xyz;
		vs_out.normal = normalize(mat3(view) * normal);
		vs_out.light = normalize(view * vec4(light.position, 1.0)).xyz;
		vs_out.uv = uv;

		gl_Position = projection * view * vec4(position, 1.0);
    })vs";

#if !DEBUG_SHOW_NORMALS
	//
	// FS mostly from https://learnopengl.com/PBR/Lighting
	// 
	static const char* fs_source = R"fs(
	#version 330
	#define PI 3.1415926538

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

	struct Light
	{
		vec3 position;
		vec3 color;
		float constant;
		float linear;
		float quadratic;
	};
	uniform Light light;

	out vec4 out_color;

	float Pow5(float x)
	{
		float xx = x * x;
		return xx * xx * x;
	}

	float DistributionGGX(float NdotH, float roughness)
	{
		float a      = roughness * roughness;
		float a2     = a * a;
		float NdotH2 = NdotH * NdotH;
	
		float num   = a2;
		float denom = (NdotH2 * (a2 - 1.0) + 1.0);
		denom = PI * denom * denom;
	
		return num / denom;
	}

	float GeometrySchlickGGX(float NdotV, float roughness)
	{
		float r = (roughness + 1.0);
		float k = (r * r) / 8.0;

		float num   = NdotV;
		float denom = NdotV * (1.0 - k) + k;
	
		return num / denom;
	}

	float GeometrySmith(float NdotV, float NdotL, float roughness)
	{
		float ggx2 = GeometrySchlickGGX(NdotV, roughness);
		float ggx1 = GeometrySchlickGGX(NdotL, roughness);
	
		return ggx1 * ggx2;
	}

	vec3 FresnelSchlick(float HdotV, vec3 F0)
	{
		return F0 + (1.0 - F0) * Pow5(1.0 - HdotV);
	}  

	void main()
	{
		vec3 n = normalize(vs_in.normal);
		vec3 v = normalize(/* view */ - vs_in.position);
		vec3 l = normalize(vs_in.light - vs_in.position);
		vec3 h = normalize(l + v);

		float NdotL = max(dot(n, l), 0.0);
		float NdotH = max(dot(n, h), 0.0);
		float NdotV = max(dot(n, v), 0.0);
		float VdotH = max(dot(v, h), 0.0);

		vec3 F0 = vec3(0.04); 
		F0 = mix(F0, material.albedo, material.metalness);

		float N = DistributionGGX(NdotH, material.roughness);
		float G = GeometrySmith(NdotV, NdotL, material.roughness);      
        vec3  F = FresnelSchlick(VdotH, F0);

		vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - material.metalness;
        
        vec3 numerator    = N * G * F;
        float denominator = 4.0 * NdotV * NdotL;
        vec3 specular     = numerator / max(denominator, 0.001);

		float distance    = length(l);
        float attenuation = 1.0 / (light.constant +
		                           light.linear * distance +
		                           light.quadratic * (distance * distance));
        vec3 radiance     = light.color * attenuation;
		
		vec3 direct = vec3(kD * material.albedo / PI + specular) * radiance * NdotL;
		vec3 indirect = vec3(0, 0, 0);

		out_color = vec4(direct + indirect, 1.0);
	}
	)fs";

#else

	//
	// debug FS
	// 
	static const char* fs_source = R"fs(
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
