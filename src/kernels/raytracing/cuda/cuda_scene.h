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
    unsigned int size_;
    unsigned int capacity_;
    T* buffer_;

public:
    __device__ explicit CUDAVector(unsigned int capacity = 0)
        : size_(0)
        , capacity_(capacity)
        , buffer_(static_cast<T*>(::operator new(capacity_ * sizeof(T))))
    {
    }

    __device__ ~CUDAVector()
    {
        for (unsigned int i = 0; i < size_; ++i)
        {
            buffer_[i].~T();
        }
        ::operator delete(buffer_);
    }

    __device__ CUDAVector(const CUDAVector& other)
        : size_(0)
        , capacity_(other.capacity_)
        , buffer_(static_cast<T*>(::operator new(capacity_ * sizeof(T))))
    {
        for (unsigned int i = 0; i < other.size_; ++i)
        {
            push_back(other.buffer_[i]);
        }
    }

    __device__ CUDAVector& operator=(CUDAVector& other)
    {
        if (this != other)
        {
            CUDAVector<T> temp(other);
            other.swap(*this);
        }
        return *this;
    }

    __device__ CUDAVector(CUDAVector&& other) noexcept
        : size_(0)
        , capacity_(0)
        , buffer_(nullptr)
    {
        if (this != other)
        {
            other.swap(*this);
        }
    }

    __device__ CUDAVector& operator=(CUDAVector&& other) noexcept
    {
        other.swap(*this);
        return *this;
    }

    __device__ T& operator[](unsigned int index)
    {
        return buffer_[index];
    }

    __device__ const T& operator[](unsigned int index) const
    {
        return buffer_[index];
    }

    __device__ T* begin() const
    {
        return buffer_;
    }

    __device__ T* end() const
    {
        return buffer_ + size_;
    }

    __device__ unsigned int size() const
    {
        return size_;
    }

    __device__ bool empty() const
    {
        return size_ == 0;
    }

    __device__ void push_back(const T& value)
    {
        if (size_ == capacity_)
        {
            capacity_ = capacity_ * 2 + 1;
            realloc();
        }
        
        new (buffer_ + size_++) T(value);
    }

    __device__ void pop_back()
    {
        buffer_[size_--].~T();
    }

    __device__ const T& front() const
    {
        return buffer_[0];
    }

    __device__ const T& back() const
    {
        return buffer_[size_ - 1];
    }

    __device__ void assign(const T* first, const T* last)
    {
        while (first != last)
            push_back(*first++);
    }

    __device__ void clear()
    {
        for (unsigned int i = 0; i < size_; ++i)
        {
            pop_back();
        }
    }

    __device__ void swap(CUDAVector& other) noexcept
    {
        _swap(capacity_, other.capacity_);
        _swap(size_, other.size_);
        _swap(buffer_, other.buffer_);
    }

private:

    template<typename AnyType>
    __device__ void _swap(AnyType& a, AnyType& b)
    {
        AnyType temp(std::move(a));
        a = std::move(b);
        b = std::move(temp);
    }

    __device__ void realloc()
    {
        CUDAVector<T> expanded(capacity_);
        for (unsigned int i = 0; i < size_; ++i)
        {
            expanded.push_back(buffer_[i]);
        }
        expanded.swap(*this);
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
    accel::Node<CUDATriangle, CUDAVector>* h_scenetree;
    accel::Node<CUDATriangle, CUDAVector>* d_scenetree;
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
