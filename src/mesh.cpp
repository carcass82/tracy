/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "mesh.h"
#include <cfloat>

void Mesh::ComputeNormals()
{
	for (int i = 0; i < static_cast<int>(indices_.size()); i += 3)
	{
		Vertex& v1 = vertices_[indices_[i + 0]];
		Vertex& v2 = vertices_[indices_[i + 1]];
		Vertex& v3 = vertices_[indices_[i + 2]];

		vec3 normal = normalize(cross(v2.pos - v1.pos, v3.pos - v1.pos));
		v1.normal = v2.normal = v3.normal = normal;
	}
}

void Mesh::ComputeTangentsAndBitangents()
{
	// Lengyel, Eric. "Computing Tangent Space Basis Vectors for an Arbitrary Mesh"
	// http://www.terathon.com/code/tangent.html
	
	for (int i = 0; i < static_cast<int>(indices_.size()); i += 3)
	{
		Vertex& v1 = vertices_[indices_[i + 0]];
		Vertex& v2 = vertices_[indices_[i + 1]];
		Vertex& v3 = vertices_[indices_[i + 2]];

		// Edges of the triangle : position delta
		vec3 delta_pos1 = v2.pos - v1.pos;
		vec3 delta_pos2 = v3.pos - v1.pos;

		// UV delta
		vec2 delta_uv1 = v2.uv0 - v1.uv0;
		vec2 delta_uv2 = v3.uv0 - v1.uv0;

		float r = 1.f / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
		vec3 tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
		vec3 bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * r;

		v1.tangent += v2.tangent = v3.tangent = tangent;
		v1.bitangent += v2.bitangent = v3.bitangent = bitangent;
	}

	for (Vertex& v : vertices_)
	{
		// orthonormalize tangent
		v.tangent = normalize(v.tangent - v.normal * dot(v.normal, v.tangent));

		// ensure coherent handedness
		float sign = (dot(cross(v.normal, v.tangent), v.bitangent) < 0.0f) ? -1.0f : 1.0f;
		v.bitangent = sign * cross(v.normal, v.tangent);
	}
}

void Mesh::ComputeBoundingBox()
{
	vec3 vmin(FLT_MAX);
	vec3 vmax(-FLT_MAX);
	for (const Vertex& vertex : vertices_)
	{
		vmin = pmin(vmin, vertex.pos);
		vmax = pmax(vmax, vertex.pos);
	}

	center_ = { (vmax + vmin) / 2.f };
	size_ = { vmax - vmin };
	aabb_ = { vmin, vmax };
}
