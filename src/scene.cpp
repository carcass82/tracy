/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "scene.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "ext/tiny_obj_loader.h"

constexpr inline uint32_t make_id(char a, char b, char c = '\0', char d = '\0')
{
	return a | b << 8 | c << 16 | d << 24;
}

Mesh& Scene::AddSphere(const vec3& in_center, float in_radius, uint32_t steps /* = 32 */)
{
	vector<Vertex> vertices;
	vector<Index> indices;

	for (uint32_t lon = 0; lon < steps; ++lon)
	{
		float delta_theta1 = (float)lon / steps;
		float delta_theta2 = (float)(lon + 1) / steps;
		float theta1 = delta_theta1 * PI;
		float theta2 = delta_theta2 * PI;

		for (uint32_t lat = 0; lat < steps; ++lat)
		{
			float delta_phi1 = (float)lat / steps;
			float delta_phi2 = (float)(lat + 1) / steps;
			float phi1 = delta_phi1 * 2.f * PI;
			float phi2 = delta_phi2 * 2.f * PI;

			// phi2  phi1
			//  |     |
			//  2-----1 -- theta1
			//  |\    |
			//  | \   |
			//  |  \  |
			//  |   \ |
			//  3-----4 -- theta2

			//vertex1 = vertex on a sphere of radius r at spherical coords theta1, phi1
			vec3 pos = vec3{ sinf(theta1) * cosf(phi1), sinf(theta1) * sinf(phi1), cosf(theta1) };
			vec2 uv = vec2{ delta_phi1, delta_theta1 };
			vertices.emplace_back(in_center + pos * in_radius, pos, uv);

			//vertex2 = vertex on a sphere of radius r at spherical coords theta1, phi2
			pos = vec3{ sinf(theta1) * cosf(phi2), sinf(theta1) * sinf(phi2), cosf(theta1) };
			uv = vec2{ delta_phi2, delta_theta1 };
			vertices.emplace_back(in_center + pos * in_radius, pos, uv);

			//vertex3 = vertex on a sphere of radius r at spherical coords theta2, phi2
			pos = vec3{ sinf(theta2) * cosf(phi2), sinf(theta2) * sinf(phi2), cosf(theta2) };
			uv = vec2{ delta_phi2, delta_theta2 };
			vertices.emplace_back(in_center + pos * in_radius, pos, uv);

			//vertex4 = vertex on a sphere of radius r at spherical coords theta2, phi1
			pos = vec3{ sinf(theta2) * cosf(phi1), sinf(theta2) * sinf(phi1), cosf(theta2) };
			uv = vec2{ delta_phi1, delta_theta2 };
			vertices.emplace_back(in_center + pos * in_radius, pos, uv);

			uint32_t baseidx = static_cast<uint32_t>(vertices.size()) - 4;

			if (lon == 0) // top cap
			{
				indices.push_back(baseidx + 0);
				indices.push_back(baseidx + 3);
				indices.push_back(baseidx + 2);
			}
			else if (lon + 1 == steps) //end cap
			{
				indices.push_back(baseidx + 2);
				indices.push_back(baseidx + 1);
				indices.push_back(baseidx + 0);
			}
			else
			{
				indices.push_back(baseidx + 0);
				indices.push_back(baseidx + 3);
				indices.push_back(baseidx + 1);

				indices.push_back(baseidx + 1);
				indices.push_back(baseidx + 3);
				indices.push_back(baseidx + 2);
			}
		}
	}

	for (uint32_t i = 0; i < vertices.size(); ++i)
	{
		vertices[i].normal = normalize(vertices[i].normal);
	}

	return objects_.emplace_back(vertices, indices).ComputeBoundingBox().ComputeTangentsAndBitangents();
}

Mesh& Scene::AddBox(const vec3& bottom, const vec3& top)
{
	const vec3 vertices[] = {
		{ top.x,    top.y,    bottom.z },
		{ top.x,    bottom.y, bottom.z },
		{ top.x,    top.y,    top.z    },
		{ top.x,    bottom.y, top.z    },
		{ bottom.x, top.y,    bottom.z },
		{ bottom.x, bottom.y, bottom.z },
		{ bottom.x, top.y,    top.z    },
		{ bottom.x, bottom.y, top.z    }
	};

	//
	// TODO: set UVs
	//
	//const vec2 uv[] = {
	//	vec2{ 0.0f, 0.0f },
	//	vec2{ 1.0f, 0.0f },
	//	vec2{ 1.0f, 1.0f },
	//	vec2{ 0.0f, 1.0f }
	//};

	const vec3 normals[] = {
		vec3{  0.0f,  1.0f,  0.0f },
		vec3{  0.0f,  0.0f,  1.0f },
		vec3{ -1.0f,  0.0f,  0.0f },
		vec3{  0.0f, -1.0f,  0.0f },
		vec3{  1.0f,  0.0f,  0.0f },
		vec3{  0.0f,  0.0f, -1.0f } 
	};

	vector<Vertex> boxvertices;
	vector<Index> boxindices;

	boxvertices.emplace_back(vertices[4], normals[0]); // 0
	boxvertices.emplace_back(vertices[2], normals[0]); // 1
	boxvertices.emplace_back(vertices[0], normals[0]); // 2
	boxindices.push_back(0); boxindices.push_back(1); boxindices.push_back(2);

	boxvertices.emplace_back(vertices[2], normals[1]); // 3
	boxvertices.emplace_back(vertices[7], normals[1]); // 4
	boxvertices.emplace_back(vertices[3], normals[1]); // 5
	boxindices.push_back(3); boxindices.push_back(4); boxindices.push_back(5);

	boxvertices.emplace_back(vertices[6], normals[2]); // 6
	boxvertices.emplace_back(vertices[5], normals[2]); // 7
	boxvertices.emplace_back(vertices[7], normals[2]); // 8
	boxindices.push_back(6); boxindices.push_back(7); boxindices.push_back(8);

	boxvertices.emplace_back(vertices[1], normals[3]); // 9
	boxvertices.emplace_back(vertices[7], normals[3]); // 10
	boxvertices.emplace_back(vertices[5], normals[3]); // 11
	boxindices.push_back(9); boxindices.push_back(10); boxindices.push_back(11);

	boxvertices.emplace_back(vertices[0], normals[4]); // 12
	boxvertices.emplace_back(vertices[3], normals[4]); // 13
	boxvertices.emplace_back(vertices[1], normals[4]); // 14
	boxindices.push_back(12); boxindices.push_back(13); boxindices.push_back(14);

	boxvertices.emplace_back(vertices[4], normals[5]); // 15
	boxvertices.emplace_back(vertices[1], normals[5]); // 16
	boxvertices.emplace_back(vertices[5], normals[5]); // 17
	boxindices.push_back(15); boxindices.push_back(16); boxindices.push_back(17);

	boxvertices.emplace_back(vertices[6], normals[0]); // 18
	boxindices.push_back(0); boxindices.push_back(18); boxindices.push_back(1);

	boxvertices.emplace_back(vertices[6], normals[1]); // 19
	boxindices.push_back(3); boxindices.push_back(19); boxindices.push_back(4);

	boxvertices.emplace_back(vertices[4], normals[2]); // 20
	boxindices.push_back(6); boxindices.push_back(20); boxindices.push_back(7);

	boxvertices.emplace_back(vertices[3], normals[3]); // 21
	boxindices.push_back(9); boxindices.push_back(21); boxindices.push_back(10);

	boxvertices.emplace_back(vertices[2], normals[4]); // 22
	boxindices.push_back(12); boxindices.push_back(22); boxindices.push_back(13);

	boxvertices.emplace_back(vertices[0], normals[5]); // 23
	boxindices.push_back(15); boxindices.push_back(23); boxindices.push_back(16);

	return objects_.emplace_back(boxvertices, boxindices).ComputeBoundingBox().ComputeTangentsAndBitangents();
}

Mesh& Scene::AddTriangle(const vec3& v1, const vec3& v2, const vec3& v3)
{
	vector<Vertex> v{ v1, v2, v3 };
	vector<Index> i{ 0, 1, 2 };
	
	return objects_.emplace_back(v, i).ComputeBoundingBox().ComputeNormals().ComputeTangentsAndBitangents();
}

Mesh& Scene::AddMesh(Mesh&& mesh, bool compute_normals /* = false */)
{
	if (compute_normals)
	{
		return objects_.emplace_back(std::move(mesh)).ComputeBoundingBox().ComputeNormals().ComputeTangentsAndBitangents();
	}
	else
	{
		return objects_.emplace_back(std::move(mesh)).ComputeBoundingBox().ComputeTangentsAndBitangents();
	}
}

bool Scene::Init(const char* scene_path, uint32_t& inout_width, uint32_t& inout_height)
{
	constexpr uint32_t ID_SCN = make_id('S', 'C', 'N', '\0');
	constexpr uint32_t ID_OUT = make_id('O', 'U', 'T', '\0');
	constexpr uint32_t ID_CAM = make_id('C', 'A', 'M', '\0');
	constexpr uint32_t ID_MTL = make_id('M', 'T', 'L', '\0');
	constexpr uint32_t ID_SKY = make_id('S', 'K', 'Y', '\0');
	constexpr uint32_t ID_OBJ = make_id('O', 'B', 'J', '\0');
	constexpr uint32_t ID_TRI = make_id('T', 'R', 'I', '\0');

	if (FILE* fp = fopen(scene_path, "r"))
	{
		TracyLog("reading from scene file '%s'\n", scene_path);
		static char line[512];

		while (fgets(line, 512, fp))
		{
			if (line[0] == '#' || line[0] == '\n')
			{
				continue;
			}

			static char params[512];
			char id[3];
			if (sscanf(line, "%c%c%c %[^\n]", &id[0], &id[1], &id[2], params) == 4)
			{
				uint32_t uid = make_id(id[0], id[1], id[2], '\0');

				switch (uid)
				{
				case ID_SCN:
					TracyLog("found SCN marker, good!\n");
					scene_name_ = params;
					break;

				case ID_OUT:
					TracyLog("found OUT params: %s\n", params);
					{
						int w, h;
						if (sscanf(params, "%d %d", &w, &h) == 2)
						{
							inout_width = w;
							inout_height = h;
						}
					}
					break;

				case ID_CAM:
					TracyLog("found CAM: %s\n", params);
					{
						vec3 eye, center, up;
						float fov;
						float ratio = float(inout_width) / float(max(inout_height, 1u));

						if (sscanf(params, "(%f,%f,%f) (%f,%f,%f) (%f,%f,%f) %f", &eye.x, &eye.y, &eye.z,
						                                                          &center.x, &center.y, &center.z,
						                                                          &up.x, &up.y, &up.z,
						                                                          &fov) == 10)
						{
							camera_.Setup(eye, center, up, fov, ratio);
						}
					}
					break;

				case ID_MTL:
					TracyLog("found MTL: %s\n", params);
					{
						char mat_name[16];
						char mat_type;

						vec3 albedo;
						float param = .0f;

						int num = sscanf(params, "%s %c (%f,%f,%f) %f", mat_name,
						                                                &mat_type,
						                                                &albedo.x, &albedo.y, &albedo.z,
						                                                &param);
						{
							Material::MaterialID material_type;
							switch (mat_type)
							{
							case 'E':
								material_type = Material::eEMISSIVE;
								break;
							case 'L':
								material_type = Material::eLAMBERTIAN;
								break;
							case 'M':
								material_type = Material::eMETAL;
								break;
							case 'D':
								material_type = Material::eDIELECTRIC;
								break;
							default:
								material_type = Material::eINVALID;
								break;
							}

							float roughness = (num == 6) ? param : .0f;
							float ior = (num == 6) ? param : 1.f;

							materials_[mat_name] = Material(material_type, albedo, roughness, ior);
						}
					}
					break;

				case ID_SKY:
					TracyLog("found SKY: %s\n", params);
					{
						vec3 albedo;
						if (sscanf(params, "(%f,%f,%f)", &albedo.x, &albedo.y, &albedo.z) == 3)
						{
							materials_[SKY_MATERIAL_NAME] = Material(Material::eEMISSIVE, albedo);
						}
					}
					break;


				case ID_OBJ:
					TracyLog("found OBJ: %s\n", params);
					{
						char obj_type;
						char subparams[64];
						if (sscanf(params, "%c %[^\n]", &obj_type, subparams) == 2)
						{
							switch (obj_type)
							{
							case 'S':
							{
								vec3 center;
								float radius;

								char mat_name[16];
								if (sscanf(subparams, "(%f,%f,%f) %f %s", &center.x, &center.y, &center.z, &radius, mat_name) == 5)
								{
									AddSphere(center, radius).SetMaterial(&materials_[mat_name]);
								}
							}
							break;

							case 'B':
							{
								vec3 min_box;
								vec3 max_box;

								char mat_name[16];
								if (sscanf(subparams, "(%f,%f,%f) (%f,%f,%f) %s", &min_box.x, &min_box.y, &min_box.z,
									&max_box.x, &max_box.y, &max_box.z,
									mat_name) == 7)
								{
									AddBox(min_box, max_box).SetMaterial(&materials_[mat_name]);
								}
							}
							break;

							case 'T':
							{
								vec3 v1, v2, v3;

								char mat_name[16];
								if (sscanf(subparams, "(%f,%f,%f) (%f,%f,%f) (%f,%f,%f) %s", &v1.x, &v1.y, &v1.z,
									&v2.x, &v2.y, &v2.z,
									&v3.x, &v3.y, &v3.z,
									mat_name) == 10)
								{
									AddTriangle(v1, v2, v3).SetMaterial(&materials_[mat_name]);
								}
							}
							break;
							}
						}
					}
					break;

				case ID_TRI:
					TracyLog("found TRI: %s\n", params);
					{
						char file_name[256];
						char mat_name[16];
						if (sscanf(params, "%s %s", file_name, mat_name) == 2)
						{
							tinyobj::attrib_t attrib;
							vector<tinyobj::shape_t> obj_shapes;
							vector<tinyobj::material_t> obj_materials;
							string warn, err;

							if (tinyobj::LoadObj(&attrib, &obj_shapes, &obj_materials, &warn, &err, file_name))
							{
								for (const tinyobj::shape_t& shape : obj_shapes)
								{
									vector<Vertex> m_Vertices;
									vector<Index> m_Indices;
									bool recompute_normals = true;
									unordered_map<int, int> indices_remap;

									for (const tinyobj::index_t& index : shape.mesh.indices)
									{
										if (indices_remap.count(index.vertex_index) > 0)
										{
											m_Indices.emplace_back(indices_remap[index.vertex_index]);
											continue;
										}

										const int voffset = 3 * index.vertex_index;
										vec3 pos = vec3{ attrib.vertices[voffset], attrib.vertices[voffset + 1], attrib.vertices[voffset + 2] };

										vec3 normal;
										const int noffset = 3 * index.normal_index;
										if (index.normal_index != -1)
										{
											recompute_normals = false;
											normal = vec3{ attrib.normals[noffset], attrib.normals[noffset + 1], attrib.normals[noffset + 2] };
										}

										vec2 uv;
										const int uvoffset = 2 * index.texcoord_index;
										if (index.texcoord_index != -1)
										{
											uv = vec2{ attrib.texcoords[uvoffset], attrib.vertices[uvoffset + 1] };
										}

										m_Vertices.emplace_back(pos, normal, uv);

										int last_inserted = (int)m_Vertices.size() - 1;
										m_Indices.emplace_back(last_inserted);
										indices_remap[index.vertex_index] = last_inserted;
									}

									AddMesh({m_Vertices, m_Indices}, recompute_normals).SetMaterial(&materials_[mat_name]);
								}
							}
						}
					}
					break;

				default:
					TracyLog("unsupported: %s\n", line);
					break;

				}
			}
		}
		fclose(fp);

		// create default black sky material
		if (materials_.count(SKY_MATERIAL_NAME) == 0)
		{
			materials_[SKY_MATERIAL_NAME] = Material(Material::eEMISSIVE, {});
		}

		return true;
	}

	return false;
}
