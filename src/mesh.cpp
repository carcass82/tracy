/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "mesh.h"
#include <cfloat>

Mesh& Mesh::ComputeNormals()
{
	for (int i = 0; i < static_cast<int>(indices_.size()); i += 3)
	{
		Vertex& v1 = vertices_[indices_[i + 0]];
		Vertex& v2 = vertices_[indices_[i + 1]];
		Vertex& v3 = vertices_[indices_[i + 2]];

		vec3 normal = normalize(cross(v2.pos - v1.pos, v3.pos - v1.pos));
		v1.normal = v2.normal = v3.normal = normal;
	}

	return *this;
}

Mesh& Mesh::ComputeBoundingBox()
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

	return *this;
}
