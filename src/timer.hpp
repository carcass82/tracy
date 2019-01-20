/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include <chrono>
using namespace std::chrono_literals;
using std::chrono::high_resolution_clock;

class Timer
{
public:
    Timer() : span(0ms) {}

    void begin()             { t0 = high_resolution_clock::now(); }
    void end()               { t1 = high_resolution_clock::now(); span += (t1 - t0); }
    double duration() const  { return span.count() / 1000.0; }
    void reset()             { span = 0ms; }

private:
    high_resolution_clock::time_point t0;
    high_resolution_clock::time_point t1;
    std::chrono::duration<double, std::milli> span;
};
