/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once

#if !defined(fastrand)

#if RANDOM_XORSHIFT

//
// PRNG from
// https://en.wikipedia.org/wiki/Xorshift
// (Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs")
//
inline RandomCtxData initrand()
{
    return 0xABCDEFu;
}

inline float fastrand(RandomCtx ctx)
{
    u32 x{ ctx };
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    ctx = x;

    return x / static_cast<float>(UINT32_MAX);
}

#elif RANDOM_LCG

//
// PRNG from
// https://software.intel.com/en-us/articles/fast-random-number-generator-on-the-intel-pentiumr-4-processor/
// with constant values from glibc (according to https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use)
//
inline RandomCtxData initrand()
{
    return 0xABCDEFu;
}

inline float fastrand(RandomCtx ctx)
{
    static constexpr u32 multiplier{ 1103515245u };
    static constexpr u32 increment{ 12345u };

    // Advance internal state
    u32 x{ ctx * multiplier + increment };

    ctx = x;

    return x / static_cast<float>(UINT32_MAX);
}

#elif RANDOM_PCG

//
// PRNG from https://www.pcg-random.org/download.html
// updated to exploit instruction level parallelism by reducing data dependencies
// (see https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering)
//
inline RandomCtxData initrand()
{
    return 0xABCDEFu;
}

inline float fastrand(RandomCtx ctx)
{
    u32 state = ctx;
    ctx = ctx * 747796405u + 2891336453u;
    u32 word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return ((word >> 22u) ^ word) / static_cast<float>(UINT32_MAX);
}

#else

inline RandomCtxData initrand()
{
    srand(0xABCDEFu);
    return 0u;
}

//
// Default rand() [0...1]
//
inline float fastrand(RandomCtx ctx)
{ 
    return rand() / static_cast<float>(RAND_MAX);
}

#endif

#endif
