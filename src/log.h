/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "common.h"
#include <cstdio>
#include <cstdarg>

inline void TracyLog(const char* msg, ...)
{
    constexpr uint32_t MAX_BUFFER_SIZE{ 1024 };
    static char buffer[MAX_BUFFER_SIZE]{};

    va_list args;
    va_start(args, msg);
    vsnprintf(buffer, MAX_BUFFER_SIZE, msg, args);
    va_end(args);

#if defined(_WIN32)
    OutputDebugStringA(buffer);
#else
    fputs(buffer, stderr);
#endif
}
