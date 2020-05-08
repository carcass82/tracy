/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include <vector>
using std::vector;

#include <string>
using std::string;

#include <unordered_map>
using std::unordered_map;

#include "log.h"
#include "material.h"
#include "mesh.h"
#include "camera.h"

class Scene
{
public:

	Mesh& AddSphere(const vec3& in_center, float in_radius, int steps = 32);

	Mesh& AddBox(const vec3& bottom, const vec3& top);

	Mesh& AddTriangle(const vec3& v1, const vec3& v2, const vec3& v3);

	Mesh& AddMesh(Mesh&& mesh, bool compute_normals = false);

	bool Init(const char* scene_path, int& width, int& height);

	Camera& GetCamera()                    { return camera_; }

	const Camera& GetCamera() const        { return camera_; }

	const vector<Mesh>& GetObjects() const { return objects_; }

	const Mesh& GetObject(int i) const     { return objects_[i]; }

	const string& GetName() const          { return scene_name_; }

	uint32_t GetObjectCount() const        { return static_cast<uint32_t>(objects_.size()); }

	uint32_t GetTriCount() const;
	
	const Material* GetSkyMaterial() const
	{
		if (UNLIKELY(!sky_material_))
		{
			sky_material_ = &materials_.at(SKY_MATERIAL_NAME);
		}

		return sky_material_;
	}

private:
	Camera camera_;
	vector<Mesh> objects_;
	unordered_map<string, Material> materials_;
	mutable const Material* sky_material_ = {};
	string scene_name_;

	const char* SKY_MATERIAL_NAME = "__sky__";
};

//
// --------------------------------------------------------------------------
//

inline uint32_t Scene::GetTriCount() const
{
	uint32_t res = 0;
	for (const auto& object : objects_)
	{
		res += object.GetTriCount();
	}

	return res;
}
