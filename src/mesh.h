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
#include "vertex.h"
#include "aabb.h"
#include "material.h"


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

	void ComputeNormals();

	void ComputeTangentsAndBitangents();

	void ComputeBoundingBox();

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
