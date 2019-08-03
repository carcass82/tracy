/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <assert.h>
#include <time.h>
#include <thread>
#include <omp.h>

#if USE_GLM
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/fast_trigonometry.hpp>
constexpr float PI = 3.1415926535897932f;
using glm::vec3;
using glm::vec2;
using glm::max;
using glm::min;
using glm::clamp;
using glm::radians;
using glm::lerp;
template<typename T> constexpr inline void swap(T& a, T& b) { T tmp(a); a = b; b = tmp; }
#define atan2f(x, y) glm::fastAtan(x, y)
#define sinf(x) glm::fastSin(x)
#define cosf(x) glm::fastCos(x)
#else
#include "ext/cclib/cclib.h"
using cc::math::PI;
using cc::math::mat4;
using cc::math::vec4;
using cc::math::vec3;
using cc::math::vec2;
using cc::util::max;
using cc::util::min;
using cc::util::clamp;
using cc::util::array_size;
using cc::math::radians;
using cc::math::lerp;
using cc::math::dot;
using cc::util::swap;
using cc::math::lookAt;
using cc::math::perspective;
using cc::math::inverse;
#define atan2f(x, y) cc::math::fast::atan2f(x, y)
#define sinf(x) cc::math::fast::sinf(x)
#define cosf(x) cc::math::fast::cosf(x)
#endif

#if defined(USE_CUDA)
extern "C" void cuda_setup(const char* /* path */, int /* w */, int /* h */);
extern "C" void cuda_trace(int /* w */, int /* h */, int /* ns */, float* /* output */, int& /* totrays */);
extern "C" void cuda_cleanup();
#endif

#include "timer.hpp"
#include "geom.hpp"
#if !defined(USE_CUDA)
#include "ray.hpp"
#include "textures/texture.hpp"
#include "materials/material.hpp"
#include "shapes/shape.hpp"
#include "camera.hpp"
#include "scenes.hpp"


extern "C" void setup(const char* path, Camera& cam, float aspect, IShape** world)
{
    Scene scene = load_scene(path, aspect);

    cam = scene.cam;
    *world = scene.world;
}

// max "bounces" for tracing
#ifndef MAX_DEPTH
 #define MAX_DEPTH 5
#endif
vec3 color(const Ray& r, IShape* world, int depth, int& raycount)
{
#if CPU_RECURSIVE // first, recursive implementation

    ++raycount;

    HitData rec;
    if (world->hit(r, 1e-3f, FLT_MAX, rec))
    {
        //
        // debug - show normals
        //
        //return .5f * normalize((1.f + rec.normal));

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

#else // iterative version, more similar to gpu and easier to read

    Ray current_ray = Ray{ r };
    vec3 current_color = vec3{ 1.f, 1.f, 1.f };

    for (int i = 0; i < MAX_DEPTH; ++i)
    {
        ++raycount;

        HitData rec;
        if (world->hit(current_ray, 1e-3f, FLT_MAX, rec))
        {
            //
            // debug - show normals
            //
            //return .5f * normalize((1.f + rec.normal));
            
            ScatterData srec;
            if (rec.material->scatter(current_ray, rec, srec))
            {
                current_color *= srec.attenuation;
                current_ray = srec.scattered;
            }
            else
            {
                current_color *= rec.material->emitted(current_ray, rec, rec.uv, rec.point);
                return current_color;
            }
        }
        else
        {
            return vec3{};
        }
    }

#endif

    //
    // gradient
    //
    //static const vec3 WHITE{ 1.f, 1.f, 1.f };
    //static const vec3 SKYISH{ .5f, .7f, 1.f };
    //float t = (normalize(r.get_direction()).y + 1.f) * .5f;
    //return lerp(WHITE, SKYISH, t);

    return vec3{};
}


extern "C" void trace(Camera& cam, IShape* world, int nx, int ny, int ns, vec3* output, int& totrays, size_t& pixel_idx)
{
    // ensure output buffer is properly zeroed
    for (int i = 0; i < nx * ny; ++i) output[i] = {};

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
                int raycount = 0;
                vec2 uv{ (i + fastrand()) / float(nx), (j + fastrand()) / float(ny) };
                vec3 sampled_col = color(cam.get_ray(uv.x, uv.y), world, 0, raycount);

                #pragma omp atomic
                output[j * nx + i].r += sampled_col.r;

                #pragma omp atomic
                output[j * nx + i].g += sampled_col.g;

                #pragma omp atomic
                output[j * nx + i].b += sampled_col.b;

                #pragma omp atomic
                totrays += raycount;

                // not really interested in correctness
                pixel_idx++;
            }
        }
    }
}

#if !defined(USE_CUDA) || !defined(BUILD_GUI)
constexpr inline void put_char_sequence(const char x, int n)
{
    for (int i = 0; i < n; ++i)
    {
        putchar(x);
    }
}

void progbar(size_t total, size_t samples, size_t* value, bool* quit)
{
    constexpr int progbarsize = 78;
#if defined(_DEBUG)
    const int threadcount = 1;
#else
    const int threadcount = omp_get_max_threads();
#endif

    while (!(*quit))
    {
        float progress = min(1.f, float(*value / samples) / float(total));
        int progbar = int(progress * progbarsize);

        printf("tracing (%d threads) ... [", threadcount);
        put_char_sequence('#', progbar);
        put_char_sequence(' ', progbarsize - progbar);
        printf("] %.1f%%\r", progress * 100.f);

        std::this_thread::sleep_for(250ms);
    }

    // print full bar before quitting
    printf("tracing (%d threads) ... [", threadcount);
    put_char_sequence('#', progbarsize);
    fputs("] 100.0%\n", stdout);    
}
#endif
#endif

#if !defined(BUILD_GUI)
int main(int argc, char** argv)
{
    const int nx = 1024; // w
    const int ny = 768;  // h
    const int ns = 256;   // samples

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

    const char* scene_path = "data/default.scn";

    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        fprintf(stderr, "unable to open '%s' for writing, abort\n", filename);
        return -1;
    }

#if !defined(USE_CUDA)
    Camera cam;
    IShape *world = nullptr;
    setup(scene_path, cam, float(nx) / float(ny), &world);
    if (!world)
    {
        fputs("unable to prepare for tracing, abort\n", stderr);
        return -1;
    }
#else
    cuda_setup(scene_path, nx, ny);
#endif

    // output buffer
    vec3* output = new vec3[nx * ny];

    int totrays = 0;

    // trace!
    Timer t;
    t.begin();

#if !defined(USE_CUDA)
    // update progress bar using a separate thread
    bool quit = false;
    size_t pixel_idx = 0;
    std::thread progbar_thread(progbar, nx * ny, ns, &pixel_idx, &quit);

    trace(cam, world, nx, ny, ns, output, totrays, pixel_idx);

    quit = true;
    progbar_thread.join();

#else

    cuda_trace(nx, ny, ns, reinterpret_cast<float*>(output), totrays);

#endif

    t.end();

    printf("Average: %.2f MRays/s\n", (totrays / 1'000'000.0) / t.duration());

#if defined(USE_CUDA)
    cuda_cleanup();
#endif

    //
    // output to ppm (y inverted)
    // gamma 2.0
    //
    fprintf(fp, "P6\n%d %d %d\n", nx, ny, 0xff);
    for (int j = ny - 1; j >= 0; --j)
    {
        for (int i = 0; i < nx; ++i)
        {
            uint8_t clamped_col[] = {
                uint8_t(clamp(255.99f * sqrtf(output[j * nx + i].r / ns), 0.0f, 255.0f)),
                uint8_t(clamp(255.99f * sqrtf(output[j * nx + i].g / ns), 0.0f, 255.0f)),
                uint8_t(clamp(255.99f * sqrtf(output[j * nx + i].b / ns), 0.0f, 255.0f))
            };
            fwrite(clamped_col, sizeof(uint8_t), sizeof(clamped_col), fp);
        }
    }
    fclose(fp);

    delete[] output;

    fprintf(stderr, "finished in %.2f secs\n", t.duration());
}
#endif

extern "C" void save_screenshot(int w, int h, vec3* pbuffer)
{
    static char filename[256];
    {
        time_t t = time(nullptr);
#if !defined(USE_CUDA)
        strftime(filename, 256, "output-%Y%m%d-%H.%M.%S.ppm", localtime(&t));
#else
        strftime(filename, 256, "output-cuda-%Y%m%d-%H.%M.%S.ppm", localtime(&t));
#endif
    }

    FILE *fp = fopen(filename, "wb");
    if (fp)
    {
        // output to ppm (y inverted)
        // gamma 2.0
        fprintf(fp, "P6\n%d %d %d\n", w, h, 0xff);
        for (int j = h - 1; j >= 0; --j)
        {
            for (int i = 0; i < w; ++i)
            {
                const vec3 bitmap_col = clamp3(255.99f * sqrtf3(pbuffer[j * w + i]), .0f, 255.f);
                uint8_t clamped_col[] = { uint8_t(bitmap_col.r), uint8_t(bitmap_col.g), uint8_t(bitmap_col.b) };
                fwrite(clamped_col, sizeof(uint8_t), sizeof(clamped_col), fp);
            }
        }

        fclose(fp);
    }
}
