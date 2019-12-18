/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#if RANDOM_XORSHIFT
//
// PRNG from
// https://en.wikipedia.org/wiki/Xorshift
// (Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs")
//
inline float fastrand(unsigned int& ctx)
{
    uint32_t x = ctx;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    ctx = x;

    return (x & 0xffffff) / 16777216.0f;
}

#elif RANDOM_INTEL

//
// PRNG from
// https://software.intel.com/en-us/articles/fast-random-number-generator-on-the-intel-pentiumr-4-processor/
//
inline float fastrand(unsigned int& ctx)
{
    ctx = (214013u * ctx + 2531011u);
    return (ctx & 0xffffff) / 16777216.f;
}

#elif RANDOM_CUDA

//
// PRNG from NVIDIA Optix SDK
// <OptiX SDK>\SDK\cuda\random.h
//
CUDA_CALL inline float fastrand(unsigned int& ctx)
{
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    ctx = (LCG_A * ctx + LCG_C);

    return ((float)(ctx & 0x00ffffff) / (float)0x01000000);
}

#else

//
// Default rand() [0...1]
//
inline float fastrand(unsigned int& ctx)
{
    #pragma omp master
    {
        static bool do_init = true;
        if (do_init)
        {
            srand(ctx);
            do_init = false;
        }
    }

    return static_cast<float>(rand()) / RAND_MAX;
}

#endif
