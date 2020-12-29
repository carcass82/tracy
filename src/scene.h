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


#include "log.h"
#include "material.h"
#include "mesh.h"
#include "camera.h"

class Scene
{
public:

	static constexpr uint32_t SKY_MATERIAL_ID{ 0 };


	uint32_t AddTexture(const char* path, bool sRGB = false);

	Mesh& AddSphere(const vec3& in_center, float in_radius, uint32_t steps = 32);

	Mesh& AddBox(const vec3& bottom, const vec3& top, const mat4& transform = mat4{ 1 });

	Mesh& AddTriangle(const vec3& v1, const vec3& v2, const vec3& v3);

	Mesh& AddMesh(Mesh&& mesh, const mat4& transform = mat4{ 1 }, bool compute_normals = false);

	bool Init(const char* scene_path, uint32_t& width, uint32_t& height);

	Camera& GetCamera()                           { return camera_; }

	const Camera& GetCamera() const               { return camera_; }

	const vector<Mesh>& GetObjects() const        { return objects_; }

	const Mesh& GetObject(uint32_t i) const       { return objects_[i]; }

	const string& GetName() const                 { return scene_name_; }

	uint32_t GetObjectCount() const               { return static_cast<uint32_t>(objects_.size()); }

	uint32_t GetTriCount() const;
	
	uint32_t Width() const                        { return width_; }
										          
	uint32_t Height() const                       { return height_; }

	const Material& GetMaterial(uint32_t i) const { return materials_[i]; }

	const vector<Material>& GetMaterials() const  { return materials_; }

	const Texture& GetTexture(uint32_t i) const   { return textures_[i]; }

	const vector<Texture>& GetTextures() const    { return textures_; }


private:

	Camera camera_{};
	vector<Mesh> objects_{};
	vector<Material> materials_{ 1 };
	vector<Texture> textures_{};
	uint32_t width_{};
	uint32_t height_{};
	string scene_name_{};
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
