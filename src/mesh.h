/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include <vector>
using std::vector;

#include "common.h"
#include "aabb.h"
#include "material.h"

#if defined(CPU_KERNEL)
#include "kernels/raytracing/software/cpu_specific.h"
#elif defined(CUDA_KERNEL)
#include "kernels/raytracing/cuda/cuda_specific.h"
#elif defined(OPENGL_KERNEL)
#include "kernels/rasterization/opengl/opengl_specific.h"
#elif defined(DXR_KERNEL)
#include "kernels/raytracing/dxr/dxr_specific.h"
#else
#error "at least one module should be enabled!"
#endif

class Mesh
{
public:
	Mesh()
	{}

	Mesh(const vector<Vertex>& in_vertices, const vector<Index>& in_indices, const Material* in_material = nullptr)
		: vertices_(in_vertices)
		, indices_(in_indices)
		, material_(in_material)
	{}

	Mesh& ComputeNormals();
	
	template<typename VertexType = Vertex, bool enabled = VertexType::VertexHasTangents>
	typename std::enable_if<enabled, Mesh&>::type ComputeTangentsAndBitangents();

	template<typename VertexType = Vertex, bool enabled = VertexType::VertexHasTangents>
	typename std::enable_if<!enabled, Mesh&>::type ComputeTangentsAndBitangents();

	Mesh& ComputeBoundingBox();

	int GetVertexCount() const                    { return static_cast<int>(vertices_.size()); }

	int GetTriCount() const                       { return static_cast<int>(indices_.size()) / 3; }

	int GetIndexCount() const                     { return static_cast<int>(indices_.size()); }

	const vector<Vertex>& GetVertices() const     { return vertices_; }

	const Vertex& GetVertex(int i) const          { return vertices_[i]; }

	const vector<Index>& GetIndices() const       { return indices_; }

	const Index& GetIndex(int i) const            { return indices_[i]; }

	void SetMaterial(const Material* in_material) { material_ = in_material; }

	const Material* GetMaterial() const           { return material_; }

	const vec3& GetCenter() const                 { return center_; }

	const vec3& GetSize() const                   { return size_; }

	const BBox& GetAABB() const                   { return aabb_; }
	

private:
	vector<Vertex> vertices_;
	vector<Index> indices_;
	const Material* material_;
	vec3 center_;
	vec3 size_;
	BBox aabb_;

	friend class Scene;
};

template<typename VertexType, bool enabled>
typename std::enable_if<enabled, Mesh&>::type Mesh::ComputeTangentsAndBitangents()
{
	// Lengyel, Eric. "Computing Tangent Space Basis Vectors for an Arbitrary Mesh"
	// http://www.terathon.com/code/tangent.html

	for (int i = 0; i < static_cast<int>(indices_.size()); i += 3)
	{
		VertexType& v1 = vertices_[indices_[i + 0]];
		VertexType& v2 = vertices_[indices_[i + 1]];
		VertexType& v3 = vertices_[indices_[i + 2]];

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

	for (VertexType& v : vertices_)
	{
		// orthonormalize tangent
		v.tangent = normalize(v.tangent - v.normal * dot(v.normal, v.tangent));

		// ensure coherent handedness
		float sign = (dot(cross(v.normal, v.tangent), v.bitangent) < 0.0f) ? -1.0f : 1.0f;
		v.bitangent = sign * cross(v.normal, v.tangent);
	}

	return *this;
}

template<typename VertexType, bool enabled>
typename std::enable_if<!enabled, Mesh&>::type Mesh::ComputeTangentsAndBitangents()
{
	return *this;
}
