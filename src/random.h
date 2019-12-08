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
inline float fastrand()
{
    static uint32_t s_RndState = 123456789;
    #pragma omp threadprivate(s_RndState)

    uint32_t x = s_RndState;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    s_RndState = x;

    return (x & 0xffffff) / 16777216.0f;
}

#elif RANDOM_PCG

//
// PRNG from
// https://en.wikipedia.org/wiki/Permuted_congruential_generator
//
inline float fastrand()
{
    static uint64_t mcg_state = 0xcafef00dd15ea5e4u;
    #pragma omp threadprivate(mcg_state)
    
    uint64_t x = mcg_state;
    unsigned count = (unsigned)(x >> 61);
    mcg_state = x * 6364136223846793005u;
    x ^= x >> 22;
    
    return ((uint32_t)(x >> (22 + count)) & 0xffffff) / 16777216.f;
}

#elif RANDOM_INTEL

//
// PRNG from
// https://software.intel.com/en-us/articles/fast-random-number-generator-on-the-intel-pentiumr-4-processor/
//
inline float fastrand()
{
    static uint32_t g_state = 0xdeadbeef;
    #pragma omp threadprivate(g_state)

    g_state = (214013u * g_state + 2531011u);
    return (g_state & 0xffffff) / 16777216.f;
}

#else

//
// Default rand() [0...1]
//
#include <ctime>
inline float fastrand()
{
    #pragma omp master
    {
        static bool do_init = true;
        if (do_init)
        {
            srand(static_cast<unsigned int>(time(nullptr)));
            do_init = false;
        }
    }

    return static_cast<float>(rand()) / RAND_MAX;
}

#endif
