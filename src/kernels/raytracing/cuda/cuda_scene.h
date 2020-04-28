/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include <curand_kernel.h>

#include "camera.h"
#include "cuda_mesh.h"

#if !defined(CUDA_PREFERRED_DEVICE)
 #define CUDA_PREFERRED_DEVICE 0
#endif

#if USE_KDTREE
#include "kdtree.h"
#endif

struct CUDATriangle
{
    vec3 v[3];
    vec3 v0v1;
    vec3 v0v2;
    uint16_t mesh_idx;
    uint16_t tri_idx;

    __device__ CUDATriangle()
    {}

    __device__ CUDATriangle(const vec3& in_v0, const vec3& in_v1, const vec3& in_v2, uint16_t in_mesh, uint16_t in_triangle)
        : v{ in_v0, in_v1, in_v2 }
        , v0v1(v[1] - v[0])
        , v0v2(v[2] - v[0])
        , mesh_idx(in_mesh)
        , tri_idx(in_triangle)
    {}
};

template<typename T>
class CUDAVector
{
private:
    T* m_begin;
    unsigned int m_size;
    unsigned int m_capacity;

public:
    __device__  explicit CUDAVector()
        : m_size(0)
        , m_capacity(16)
    {
        m_begin = new T[m_capacity];
    }

    __device__ ~CUDAVector()
    {
        delete[] m_begin;
    }

    __device__ T& operator[](unsigned int index)
    {
        return m_begin[index];
    }

    __device__ const T& operator[](unsigned int index) const
    {
        return m_begin[index];
    }

    __device__ T* begin() const
    {
        return m_begin;
    }

    __device__ T* end() const
    {
        return m_begin + m_size + 1;
    }

    __device__ unsigned int size() const
    {
        return m_size;
    }

    __device__ bool empty() const
    {
        return m_size == 0;
    }

    __device__ void reserve(unsigned int newcapacity)
    {
        delete[] m_begin;
        m_begin = new T[newcapacity];

        m_capacity = newcapacity;
    }

    __device__ void push_back(const T& value)
    {
        if (!(m_size < m_capacity))
        {
            T* expanded = new T[m_capacity * 2];
            memcpy(expanded, m_begin, m_capacity * sizeof(T));
            delete[] m_begin;
            m_begin = expanded;

            m_capacity *= 2;
        }
        
        m_begin[m_size++] = value;
    }

    __device__ void pop_back()
    {
        if (m_size > 0)
        {
            --m_size;
        }
    }

    __device__ T& back() const
    {
        return m_begin[m_size - 1];
    }
};

struct CUDAScene
{
    int width;
    int height;

    vec4* d_output_;

	CUDAMesh* d_objects_;
	int objectcount_;

#if USE_KDTREE
    accel::Tree<CUDATriangle, CUDAVector>* h_scenetree;
    accel::Tree<CUDATriangle, CUDAVector>* d_scenetree;
#endif

    Material* d_sky_;

	Camera* d_camera_;

    curandState* d_rand_state;

    int h_raycount;
    int* d_raycount;

    
    int GetRayCount()
    {
        CUDAAssert(cudaMemcpy(&h_raycount, d_raycount, sizeof(int), cudaMemcpyDeviceToHost));
        return h_raycount;
    }
    
    void ResetRayCount()
    {
        CUDAAssert(cudaMemset(d_raycount, 0, sizeof(int)));
        h_raycount = 0;
    }
};
