/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

template<typename T>
class Vector
{
private:
    unsigned int size_;
    unsigned int capacity_;
    T* buffer_;

public:
    CUDA_DEVICE_CALL explicit Vector(unsigned int capacity = 0)
        : size_(0)
        , capacity_(capacity)
        , buffer_(static_cast<T*>(::operator new(capacity_ * sizeof(T))))
    {
    }

    CUDA_DEVICE_CALL ~Vector()
    {
        for (unsigned int i = 0; i < size_; ++i)
        {
            buffer_[i].~T();
        }
        ::operator delete(buffer_);
    }

    CUDA_DEVICE_CALL Vector(const Vector& other)
        : size_(0)
        , capacity_(other.capacity_)
        , buffer_(static_cast<T*>(::operator new(capacity_ * sizeof(T))))
    {
        for (unsigned int i = 0; i < other.size_; ++i)
        {
            push_back(other.buffer_[i]);
        }
    }

    CUDA_DEVICE_CALL Vector& operator=(Vector& other)
    {
        if (buffer_ != other.buffer_)
        {
            Vector<T> temp(other);
            other.swap(*this);
        }
        return *this;
    }

    CUDA_DEVICE_CALL Vector(Vector&& other) noexcept
        : size_(0)
        , capacity_(0)
        , buffer_(nullptr)
    {
        other.swap(*this);
    }

    CUDA_DEVICE_CALL Vector& operator=(Vector&& other) noexcept
    {
        other.swap(*this);
        return *this;
    }

    CUDA_DEVICE_CALL T& operator[](unsigned int index)
    {
        return buffer_[index];
    }

    CUDA_DEVICE_CALL const T& operator[](unsigned int index) const
    {
        return buffer_[index];
    }

    CUDA_DEVICE_CALL T* begin() const
    {
        return buffer_;
    }

    CUDA_DEVICE_CALL T* end() const
    {
        return buffer_ + size_;
    }

    CUDA_DEVICE_CALL unsigned int size() const
    {
        return size_;
    }

    CUDA_DEVICE_CALL bool empty() const
    {
        return size_ == 0;
    }

    CUDA_DEVICE_CALL void push_back(const T& value)
    {
        emplace_back(value);
    }

    template<typename ... Args>
    CUDA_DEVICE_CALL T& emplace_back(Args&& ... args)
    {
        if (size_ == capacity_)
        {
            capacity_ = capacity_ * 2 + 1;
            realloc();
        }

        new (buffer_ + size_) T(std::forward<Args>(args)...);
        return buffer_[size_++];
    }

    CUDA_DEVICE_CALL void pop_back()
    {
        buffer_[size_].~T();
        --size_;
    }

    CUDA_DEVICE_CALL const T& front() const
    {
        return buffer_[0];
    }

    CUDA_DEVICE_CALL const T& back() const
    {
        return buffer_[size_ - 1];
    }

    CUDA_DEVICE_CALL void clear()
    {
        for (unsigned int i = 0; i < size_; ++i)
        {
            pop_back();
        }
    }

    CUDA_DEVICE_CALL void swap(Vector& other) noexcept
    {
        SwapInternal(capacity_, other.capacity_);
        SwapInternal(size_, other.size_);
        SwapInternal(buffer_, other.buffer_);
    }

private:

    template<typename AnyType>
    CUDA_DEVICE_CALL void SwapInternal(AnyType& a, AnyType& b)
    {
        AnyType temp(std::move(a));
        a = std::move(b);
        b = std::move(temp);
    }

    CUDA_DEVICE_CALL void realloc()
    {
        Vector<T> expanded(capacity_);
        for (unsigned int i = 0; i < size_; ++i)
        {
            expanded.push_back(buffer_[i]);
        }
        expanded.swap(*this);
    }
};