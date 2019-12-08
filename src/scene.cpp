/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "scene.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "ext/tiny_obj_loader.h"

Mesh& Scene::AddSphere(const vec3& in_center, float in_radius, int steps /* = 32 */)
{
	Mesh m;

	for (int lon = 0; lon < steps; ++lon)
	{
		float delta_theta1 = (float)lon / steps;
		float delta_theta2 = (float)(lon + 1) / steps;
		float theta1 = delta_theta1 * PI;
		float theta2 = delta_theta2 * PI;

		for (int lat = 0; lat < steps; ++lat)
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

			Vertex v1;
			v1.pos = in_center + pos * in_radius;
			v1.normal = pos;
			v1.uv0 = uv;
			m.vertices_.push_back(v1);

			//vertex2 = vertex on a sphere of radius r at spherical coords theta1, phi2
			pos = vec3{ sinf(theta1) * cosf(phi2), sinf(theta1) * sinf(phi2), cosf(theta1) };
			uv = vec2{ delta_phi2, delta_theta1 };

			Vertex v2;
			v2.pos = in_center + pos * in_radius;
			v2.normal = pos;
			v2.uv0 = uv;
			m.vertices_.push_back(v2);

			//vertex3 = vertex on a sphere of radius r at spherical coords theta2, phi2
			pos = vec3{ sinf(theta2) * cosf(phi2), sinf(theta2) * sinf(phi2), cosf(theta2) };
			uv = vec2{ delta_phi2, delta_theta2 };

			Vertex v3;
			v3.pos = in_center + pos * in_radius;
			v3.normal = pos;
			v3.uv0 = uv;
			m.vertices_.push_back(v3);

			//vertex4 = vertex on a sphere of radius r at spherical coords theta2, phi1
			pos = vec3{ sinf(theta2) * cosf(phi1), sinf(theta2) * sinf(phi1), cosf(theta2) };
			uv = vec2{ delta_phi1, delta_theta2 };

			Vertex v4;
			v4.pos = in_center + pos * in_radius;
			v4.normal = pos;
			v4.uv0 = uv;
			m.vertices_.push_back(v4);

			uint32_t baseidx = static_cast<uint32_t>(m.vertices_.size()) - 4;

			if (lon == 0) // top cap
			{
				m.indices_.push_back(baseidx + 0);
				m.indices_.push_back(baseidx + 3);
				m.indices_.push_back(baseidx + 2);
			}
			else if (lon + 1 == steps) //end cap
			{
				m.indices_.push_back(baseidx + 2);
				m.indices_.push_back(baseidx + 1);
				m.indices_.push_back(baseidx + 0);
			}
			else
			{
				m.indices_.push_back(baseidx + 0);
				m.indices_.push_back(baseidx + 3);
				m.indices_.push_back(baseidx + 1);

				m.indices_.push_back(baseidx + 1);
				m.indices_.push_back(baseidx + 3);
				m.indices_.push_back(baseidx + 2);
			}
		}
	}

	for (Vertex& v : m.vertices_)
	{
		v.normal = normalize(v.normal);
	}

	m.ComputeBoundingBox();
	m.ComputeTangentsAndBitangents();

	objects_.push_back(m);
	return objects_.back();
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

	const vec2 uv[] = {
		vec2{ 0.0f, 0.0f },
		vec2{ 1.0f, 0.0f },
		vec2{ 1.0f, 1.0f },
		vec2{ 0.0f, 1.0f }
	};

	const vec3 normals[] = {
		vec3{  0.0f,  1.0f,  0.0f },
		vec3{  0.0f,  0.0f,  1.0f },
		vec3{ -1.0f,  0.0f,  0.0f },
		vec3{  0.0f, -1.0f,  0.0f },
		vec3{  1.0f,  0.0f,  0.0f },
		vec3{  0.0f,  0.0f, -1.0f } 
	};

	Mesh m;
	m.vertices_.emplace_back(vertices[4], normals[0], vec2{}, vec3{}, vec3{}); // 0
	m.vertices_.emplace_back(vertices[2], normals[0], vec2{}, vec3{}, vec3{}); // 1
	m.vertices_.emplace_back(vertices[0], normals[0], vec2{}, vec3{}, vec3{}); // 2
	m.indices_.push_back(0); m.indices_.push_back(1); m.indices_.push_back(2);

	m.vertices_.emplace_back(vertices[2], normals[1], vec2{}, vec3{}, vec3{}); // 3
	m.vertices_.emplace_back(vertices[7], normals[1], vec2{}, vec3{}, vec3{}); // 4
	m.vertices_.emplace_back(vertices[3], normals[1], vec2{}, vec3{}, vec3{}); // 5
	m.indices_.push_back(3); m.indices_.push_back(4); m.indices_.push_back(5);

	m.vertices_.emplace_back(vertices[6], normals[2], vec2{}, vec3{}, vec3{}); // 6
	m.vertices_.emplace_back(vertices[5], normals[2], vec2{}, vec3{}, vec3{}); // 7
	m.vertices_.emplace_back(vertices[7], normals[2], vec2{}, vec3{}, vec3{}); // 8
	m.indices_.push_back(6); m.indices_.push_back(7); m.indices_.push_back(8);

	m.vertices_.emplace_back(vertices[1], normals[3], vec2{}, vec3{}, vec3{}); // 9
	m.vertices_.emplace_back(vertices[7], normals[3], vec2{}, vec3{}, vec3{}); // 10
	m.vertices_.emplace_back(vertices[5], normals[3], vec2{}, vec3{}, vec3{}); // 11
	m.indices_.push_back(9); m.indices_.push_back(10); m.indices_.push_back(11);

	m.vertices_.emplace_back(vertices[0], normals[4], vec2{}, vec3{}, vec3{}); // 12
	m.vertices_.emplace_back(vertices[3], normals[4], vec2{}, vec3{}, vec3{}); // 13
	m.vertices_.emplace_back(vertices[1], normals[4], vec2{}, vec3{}, vec3{}); // 14
	m.indices_.push_back(12); m.indices_.push_back(13); m.indices_.push_back(14);

	m.vertices_.emplace_back(vertices[4], normals[5], vec2{}, vec3{}, vec3{}); // 15
	m.vertices_.emplace_back(vertices[1], normals[5], vec2{}, vec3{}, vec3{}); // 16
	m.vertices_.emplace_back(vertices[5], normals[5], vec2{}, vec3{}, vec3{}); // 17
	m.indices_.push_back(15); m.indices_.push_back(16); m.indices_.push_back(17);

	m.vertices_.emplace_back(vertices[6], normals[0], vec2{}, vec3{}, vec3{}); // 18
	m.indices_.push_back(0); m.indices_.push_back(18); m.indices_.push_back(1);

	m.vertices_.emplace_back(vertices[6], normals[1], vec2{}, vec3{}, vec3{}); // 19
	m.indices_.push_back(3); m.indices_.push_back(19); m.indices_.push_back(4);

	m.vertices_.emplace_back(vertices[4], normals[2], vec2{}, vec3{}, vec3{}); // 20
	m.indices_.push_back(6); m.indices_.push_back(20); m.indices_.push_back(7);

	m.vertices_.emplace_back(vertices[3], normals[3], vec2{}, vec3{}, vec3{}); // 21
	m.indices_.push_back(9); m.indices_.push_back(21); m.indices_.push_back(10);

	m.vertices_.emplace_back(vertices[2], normals[4], vec2{}, vec3{}, vec3{}); // 22
	m.indices_.push_back(12); m.indices_.push_back(22); m.indices_.push_back(13);

	m.vertices_.emplace_back(vertices[0], normals[5], vec2{}, vec3{}, vec3{}); // 23
	m.indices_.push_back(15); m.indices_.push_back(23); m.indices_.push_back(16);

	m.ComputeBoundingBox();
	m.ComputeTangentsAndBitangents();

	objects_.push_back(m);
	return objects_.back();
}

Mesh& Scene::AddTriangle(const vec3& v1, const vec3& v2, const vec3& v3)
{
	Mesh m;

	m.vertices_.emplace_back(v1, vec3{}, vec2{}, vec3{}, vec3{});
	m.vertices_.emplace_back(v2, vec3{}, vec2{}, vec3{}, vec3{});
	m.vertices_.emplace_back(v3, vec3{}, vec2{}, vec3{}, vec3{});
	m.indices_.emplace_back(0); m.indices_.emplace_back(1); m.indices_.emplace_back(2);

	m.ComputeBoundingBox();
	m.ComputeNormals();
	m.ComputeTangentsAndBitangents();

	objects_.push_back(m);
	return objects_.back();
}

Mesh& Scene::AddMesh(const Mesh& mesh, bool compute_normals /* = false */)
{
	Mesh m{ mesh };

	m.ComputeBoundingBox();
	if (compute_normals)
	{
		m.ComputeNormals();
	}
	m.ComputeTangentsAndBitangents();

	objects_.push_back(m);
	return objects_.back();
}

bool Scene::Init(const char* scene_path, int width, int height)
{
	constexpr uint32_t ID_SCN = make_id('S', 'C', 'N', '\0');
	constexpr uint32_t ID_CAM = make_id('C', 'A', 'M', '\0');
	constexpr uint32_t ID_MTL = make_id('M', 'T', 'L', '\0');
	constexpr uint32_t ID_OBJ = make_id('O', 'B', 'J', '\0');
	constexpr uint32_t ID_TRI = make_id('T', 'R', 'I', '\0');

	if (FILE* fp = fopen(scene_path, "r"))
	{
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
					fputs("found SCN marker, good!\n", stderr);
					scene_name_ = params;
					break;

				case ID_CAM:
					fprintf(stderr, "found cam: %s\n", params);
					{
						vec3 eye, center, up;
						float fov;
						float ratio = float(width) / float(max(height, 1));

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
					fprintf(stderr, "found mtl: %s\n", params);
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
								material_type = Material::MaterialID::eEMISSIVE;
								break;
							case 'L':
								material_type = Material::MaterialID::eLAMBERTIAN;
								break;
							case 'M':
								material_type = Material::MaterialID::eMETAL;
								break;
							case 'D':
								material_type = Material::MaterialID::eDIELECTRIC;
								break;
							default:
								material_type = Material::MaterialID::eINVALID;
								break;
							}

							float roughness = (num == 6) ? param : .0f;
							float ior = (num == 6) ? param : 1.f;

							materials_[mat_name] = Material(material_type, albedo, roughness, ior);
						}
					}
					break;


				case ID_OBJ:
					fprintf(stderr, "found obj: %s\n", params);
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
					fprintf(stderr, "found tri: %s\n", params);
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
											m_Indices.push_back(indices_remap[index.vertex_index]);
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

										m_Vertices.push_back({ pos, normal, uv, vec3{}, vec3{} });
										m_Indices.push_back((int)m_Vertices.size() - 1);
										indices_remap[index.vertex_index] = (int)m_Vertices.size() - 1;
									}

									AddMesh(Mesh{ m_Vertices, m_Indices }, recompute_normals).SetMaterial(&materials_[mat_name]);
								}
							}
						}
					}
					break;

				default:
					fprintf(stderr, "unsupported: %s\n", line);
					break;

				}
			}
		}
		fclose(fp);
		return true;
	}

	return false;
}
