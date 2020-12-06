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

    return x / static_cast<float>(UINT32_MAX);
}

#elif RANDOM_LCG

//
// PRNG from
// https://software.intel.com/en-us/articles/fast-random-number-generator-on-the-intel-pentiumr-4-processor/
// with constant values from glibc (according to https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use)
//
inline float fastrand(RandomCtx ctx)
{
    ctx = (1103515245u * ctx + 12345u);
    return ctx / static_cast<float>(UINT32_MAX);
}

#elif RANDOM_PCG

//
// PRNG from https://www.pcg-random.org/download.html
//
inline float fastrand(RandomCtx ctx)
{
    uint64_t oldstate = ctx.state;

    // Advance internal state
    ctx.state = oldstate * 6364136223846793005ULL + (ctx.inc | 1);
    
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = oldstate >> 59u;

    uint32_t rand = (xorshifted >> rot) | (xorshifted << (-(static_cast<int32_t>(rot)) & 31));

    return rand / static_cast<float>(UINT32_MAX);
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
    
    return rand() / static_cast<float>(RAND_MAX);
}

#endif

#endif
