/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#if !defined(fastrand)

#if RANDOM_XORSHIFT

//
// PRNG from
// https://en.wikipedia.org/wiki/Xorshift
// (Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs")
//
inline float fastrand(RandomCtx ctx)
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
inline float fastrand(RandomCtx ctx)
{
    ctx = (214013u * ctx + 2531011u);
    return (ctx & 0xffffff) / 16777216.f;
}

#else

//
// Default rand() [0...1]
//
inline float fastrand(RandomCtx ctx)
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

#endif
