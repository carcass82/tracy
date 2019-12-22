/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include <cstdio>
#include <cstdarg>

inline void TracyLog(const char* msg, ...)
{
    static char buffer[1024] = { 0 };

    va_list args;
    va_start(args, msg);
    vsnprintf(buffer, 1024, msg, args);
    va_end(args);

#if defined(WIN32)
    OutputDebugStringA(buffer);
#else
    fputs(buffer, stderr);
#endif
}
