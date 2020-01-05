/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include <unordered_map>
using std::unordered_map;

struct CUDAMaterial
{
	static Material* Convert(const Material* cpu_material)
	{
		if (device_materials_.count(cpu_material) == 0)
		{
			Material* d_material;
			CUDAAssert(cudaMalloc(&d_material, sizeof(Material)));
			CUDAAssert(cudaMemcpy(d_material, cpu_material, sizeof(Material), cudaMemcpyHostToDevice));

			device_materials_[cpu_material] = d_material;
		}

		return device_materials_[cpu_material];
	}

	static unordered_map<const Material*, Material*> device_materials_;
};
