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

#include "material.h"
#include "mesh.h"
#include "camera.h"

class Scene
{
public:

	Mesh& AddSphere(const vec3& in_center, float in_radius, int steps = 32);

	Mesh& AddBox(const vec3& bottom, const vec3& top);

	Mesh& AddTriangle(const vec3& v1, const vec3& v2, const vec3& v3);

	Mesh& AddMesh(const Mesh& mesh, bool compute_normals = false);

	bool Init(const char* scene_path, int& width, int& height);

	Camera& GetCamera()                    { return camera_; }

	const Camera& GetCamera() const        { return camera_; }

	const vector<Mesh>& GetObjects() const { return objects_; }

	const Mesh& GetObject(int i) const     { return objects_[i]; }

	const string& GetName() const          { return scene_name_; }

	int GetObjectCount() const             { return (int)objects_.size(); }

	int GetTriCount() const;
	
	const Material* GetSkyMaterial() const { return &materials_.at(SKY_MATERIAL_NAME); }

private:
	Camera camera_;
	vector<Mesh> objects_;
	unordered_map<string, Material> materials_;
	string scene_name_;

	const char* SKY_MATERIAL_NAME = "__sky__";
};

//
//
//

inline int Scene::GetTriCount() const
{
	int res = 0;
	for (const auto& object : objects_)
	{
		res += object.GetTriCount();
	}

	return res;
}
