/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once

#include <string>
using std::string;

#include "log.h"
#include "material.h"
#include "mesh.h"
#include "camera.h"

class Scene
{
public:

	static constexpr u32 SKY_MATERIAL_ID{ 0 };


	u32 AddTexture(const char* path, bool sRGB = false);

	Mesh& AddSphere(const vec3& in_center, float in_radius, u32 steps = 32);

	Mesh& AddBox(const vec3& bottom, const vec3& top, const mat4& transform = mat4{ 1 });

	Mesh& AddTriangle(const vec3& v1, const vec3& v2, const vec3& v3);

	Mesh& AddMesh(Mesh&& mesh, const mat4& transform = mat4{ 1 }, bool compute_normals = false);

	bool Init(const char* scene_path, u32& width, u32& height);

	Camera& GetCamera()                           { return camera_; }

	const Camera& GetCamera() const               { return camera_; }

	const vector<Mesh>& GetObjects() const        { return objects_; }

	const Mesh& GetObject(u32 i) const            { return objects_[i]; }

	const string& GetName() const                 { return scene_name_; }

	u32 GetObjectCount() const                    { return static_cast<u32>(objects_.size()); }

	u32 GetTriCount() const;
	
	u32 GetWidth() const                          { return width_; }
										          
	u32 GetHeight() const                         { return height_; }

	const Material& GetMaterial(u32 i) const      { return materials_[i]; }

	const vector<Material>& GetMaterials() const  { return materials_; }

	const Texture& GetTexture(u32 i) const        { return textures_[i]; }

	const vector<Texture>& GetTextures() const    { return textures_; }


private:

	Camera camera_{};
	vector<Mesh> objects_{};
	vector<Material> materials_{ 1 };
	vector<Texture> textures_{};
	u32 width_{};
	u32 height_{};
	string scene_name_{};
};

//
// --------------------------------------------------------------------------
//

inline u32 Scene::GetTriCount() const
{
	u32 res = 0;
	for (const auto& object : objects_)
	{
		res += object.GetTriCount();
	}

	return res;
}
