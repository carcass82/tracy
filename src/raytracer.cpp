/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 *
 *
 * TODO:
 * - read scene desc from file
 * - realtime preview of result
 */

#include <iostream>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <cfloat>
#include <cassert>
#include <ctime>
#include <thread>
#include <chrono>

#if USE_GLM
#include <glm/glm.hpp>
constexpr float PI = 3.1415926535897932f;
using glm::vec3;
using glm::vec2;
using glm::max;
using glm::min;
using glm::clamp;
using glm::radians;
template<typename T> constexpr inline void swap(T& a, T& b) { T tmp(a); a = b; b = tmp; }
#else
#include "ext/cclib/cclib.h"
using cc::math::PI;
using cc::math::vec3;
using cc::math::vec2;
using cc::util::max;
using cc::util::min;
using cc::util::clamp;
using cc::math::radians;
using cc::math::lerp;
using cc::util::swap;
using cc::util::array_size;
#define atan2f(x, y) cc::math::fast::atan2f(x, y)
#endif

#if defined(USE_CUDA)
extern "C" void cuda_trace(int, int, int, float*, int&);
#endif

#if !defined(USE_CUDA)
#include "ray.hpp"
#include "geom.hpp"
#include "textures/texture.hpp"
#include "materials/material.hpp"
#include "shapes/shape.hpp"
#include "camera.hpp"
#include "scenes.hpp"
#endif

#include "timer.hpp"

// max "bounces" for tracing
#define MAX_DEPTH 5

#if !defined(USE_CUDA)
vec3 color(const Ray& r, IShape* world, int depth, size_t& raycount)
{
    ++raycount;

    HitData rec;
    if (world->hit(r, .001f, std::numeric_limits<float>::max(), rec))
    {
        //
        // debug - show normals
        //
        //return .5f * (1.f + normalize(rec.normal));

        vec3 emitted = rec.material->emitted(r, rec, rec.uv, rec.point);

        ScatterData srec;
        if (depth < MAX_DEPTH && rec.material->scatter(r, rec, srec))
        {
            return emitted + srec.attenuation * color(srec.scattered, world, depth + 1, raycount);
        }
        else
        {
            return emitted;
        }

    }

    //
    // gradient
    //
    //static const vec3 WHITE{ 1.f, 1.f, 1.f };
    //static const vec3 SKYISH{ .5f, .7f, 1.f };
    //float t = (normalize(r.get_direction()).y + 1.f) * .5f;
    //return lerp(WHITE, SKYISH, t);

    return vec3{ .0f, .0f, .0f };
}

void progbar(size_t total, size_t samples, size_t* value, bool* quit)
{
    const size_t progbarsize = 78;

    while (!(*quit))
    {
        float progress = min(1.f, float(*value / samples) / float(total));
        int progbar = int(progress * progbarsize);

        std::cout << "tracing... ["
                  << std::string(progbar, '#')
                  << std::string(progbarsize - progbar, ' ')
                  << "] "
                  << std::fixed << std::setprecision(1) << progress * 100.f << "%\r";

        std::this_thread::sleep_for(500ms);
    }

    std::cout << "tracing... [" << std::string(progbarsize, '#') << "] 100.0%\n";
}
#endif

int main(int argc, char** argv)
{
    const int nx = 1024; // w
    const int ny = 768; // h
    const int ns = 50; // samples

#if !defined(USE_CUDA)
    Camera cam;

    // test
    //IShape* world = load_scene(eRANDOM, cam, float(nx) / float(ny));
    
    // modified cornell box
    //IShape* world = load_scene(eCORNELLBOX, cam, float(nx) / float(ny));

    // test same scene as gpu version
    IShape* world = load_scene(eTESTGPU, cam, float(nx) / float(ny));
#endif

    char filename[256];
    {
        time_t t = time(nullptr);
#if !defined(USE_CUDA)
        strftime(filename, 256, "output-%Y%m%d-%H.%M.%S.ppm", localtime(&t));
#else
        strftime(filename, 256, "output-cuda-%Y%m%d-%H.%M.%S.ppm", localtime(&t));
#endif

        if (argc == 2)
        {
            memset(filename, 0, 256);
            strncpy(filename, argv[1], 255);
        }
    }

    std::ofstream ppm_stream(filename, std::ios::binary);
    if (!ppm_stream.good())
    {
        std::cerr << "unable to open " << filename << " for writing, abort\n";
        return -1;
    }

    // output buffer
    vec3* output = new vec3[nx * ny];

    // trace!
    Timer t;
    t.begin();

    int totrays = 0;

#if !defined(USE_CUDA)
    // update progress bar using a separate thread
    bool quit = false;
    size_t pixel_idx = 0;
    std::thread progbar_thread(progbar, nx * ny, ns, &pixel_idx, &quit);

    //
    // OpenMP: collapse all 3 loops and distribute work to threads.
    //         scheduling must be dynamic to avoid work imbalance
    //         since rays could hit nothing or bounce "forever"
    //
#if defined(_MSC_VER)
 #define collapse(x) /* ms compiler does not support openmp 3.0 */
#endif

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            for (int s = 0; s < ns; ++s)
            {
                size_t raycount = 0;
                vec2 uv{ (i + fastrand()) / float(nx), (j + fastrand()) / float(ny) };
                vec3 sampled_col = color(cam.get_ray(uv.x, uv.y), world, 0, raycount);

                #pragma omp atomic
                output[j * nx + i].r += sampled_col.r;

                #pragma omp atomic
                output[j * nx + i].g += sampled_col.g;

                #pragma omp atomic
                output[j * nx + i].b += sampled_col.b;

                // not really interested in correctness
                pixel_idx++;

                #pragma omp atomic
                totrays += raycount;
            }
        }
    }

    quit = true;
    progbar_thread.join();

#else

    cuda_trace(nx, ny, ns, reinterpret_cast<float*>(output), totrays);

#endif

    t.end();

    std::cout << "Average: " << std::fixed << std::setprecision(2) << (totrays / 1'000'000.0) / t.duration() << " MRays/s\n";

    //
    // output to ppm (y inverted)
    // gamma 2.0
    //
    ppm_stream << "P6\n" << nx << " " << ny << " " << 0xff << "\n";
    for (int j = ny - 1; j >= 0; --j)
    {
        for (int i = 0; i < nx; ++i)
        {
            vec3 clamped_col = vec3{ clamp(255.99f * sqrtf(output[j * nx + i].r / ns), 0.0f, 255.0f),
                                     clamp(255.99f * sqrtf(output[j * nx + i].g / ns), 0.0f, 255.0f),
                                     clamp(255.99f * sqrtf(output[j * nx + i].b / ns), 0.0f, 255.0f) };
    
            ppm_stream << uint8_t(clamped_col.r) << uint8_t(clamped_col.g) << uint8_t(clamped_col.b);
        }
    }
    ppm_stream.close();

    delete[] output;

    std::cerr << "finished in " << std::fixed << std::setprecision(2) << t.duration() << " secs\n" << std::endl;
}
