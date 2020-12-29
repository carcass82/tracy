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
inline RandomCtxData initrand()
{
    return 0xABCDEFu;
}

inline float fastrand(RandomCtx ctx)
{
    uint32_t x{ ctx };
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
    static constexpr uint32_t multiplier{ 1103515245u };
    static constexpr uint32_t increment{ 12345u };

    // Advance internal state
    uint32_t x{ ctx * multiplier + increment };

    ctx = x;

    return x / static_cast<float>(UINT32_MAX);
}

#elif RANDOM_PCG

//
// PRNG from https://www.pcg-random.org/download.html
//
inline RandomCtxData initrand()
{
    return 0x123456789ABCDEFull;
}

inline float fastrand(RandomCtx ctx)
{
    static constexpr uint64_t multiplier{ 6364136223846793005ull };
    static constexpr uint64_t increment{ 1442695040888963407ull };

    // Advance internal state
    uint64_t x{ ctx * multiplier + increment };

    ctx = x;
    
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = static_cast<uint32_t>(((x >> 18u) ^ x) >> 27u);

    uint32_t rot = x >> 59u;
    
    uint32_t result = (xorshifted >> rot) | (xorshifted << (-(static_cast<int32_t>(rot)) & 31));

    return result / static_cast<float>(UINT32_MAX);
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
