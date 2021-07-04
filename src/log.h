/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once
#include "common.h"
#include <cstdio>
#include <cstdarg>

template <typename ... Args>
inline void TracyLog(const char* msg, Args ... args)
{
    constexpr uint32_t MAX_BUFFER_SIZE{ 1024 };
    static char buffer[MAX_BUFFER_SIZE]{};

    snprintf(buffer, MAX_BUFFER_SIZE, msg, args ...);

#if defined(_WIN32)
    OutputDebugStringA(buffer);
#else
    fputs(buffer, stderr);
#endif
}
