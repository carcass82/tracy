/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include <chrono>

class Timer
{
public:
    Timer()
        : span(0)
    {}

    void begin()
    {
        t0 = std::chrono::high_resolution_clock::now();
    }

    void end()
    {
        t1 = std::chrono::high_resolution_clock::now();
        span += std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    }

    double duration() const
    {
        return span;
    }

    void clear()
    {
        span = 0;
    }

private:
    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;
    double span;
};
