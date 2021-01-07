/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once

#include <chrono>
using namespace std::chrono_literals;
using std::chrono::high_resolution_clock;

class Timer
{
public:
    Timer()
        : span_(0ms)
    {}

    void Begin()                { t0_ = high_resolution_clock::now(); }
    void End()                  { t1_ = high_resolution_clock::now(); span_ += (t1_ - t0_); }
    double GetDuration() const  { return span_.count() * 1e-3; }
    void Reset()                { span_ = 0ms; }

private:
    high_resolution_clock::time_point t0_;
    high_resolution_clock::time_point t1_;
    std::chrono::duration<double, std::milli> span_;
};
