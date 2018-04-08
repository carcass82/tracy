/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cfloat>
#include <cassert>
#include <thread>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb_image.h"

#include "tmath.h"
#include "timer.hpp"
#include "material.hpp"
#include "camera.hpp"
#include "hitable.hpp"
#include "ray.hpp"
#include "geom.hpp"
#include "scenes.hpp"

using vmath::vec2;
using vmath::vec3;
using vutil::clamp;

vec3 color(const Ray& r, hitable* world, int depth, bool do_emissive)
{
    hit_record rec;
    if (world->hit(r, 0.01f, std::numeric_limits<float>::max(), rec)) {

        Ray scattered;
        vec3 attenuation;

        vec3 emitted = rec.mat_ptr->emitted(rec.uv, rec.p);

        if (depth < 30 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {

            // for Lambert materials, we just did explicit light (emissive) sampling and already
            // for their contribution, so if next ray bounce hits the light again, don't add
            // emission
            //
            // see https://github.com/aras-p/ToyPathTracer/commit/5b7607b89d4510623700751edb5f0837da2af23a#diff-a51e0aea7aae9c8c455717cc7d8f957bL183
            if (!do_emissive) {
                emitted = {.0f, .0f, .0f};
            }
            do_emissive = !rec.mat_ptr->islambertian();

            return emitted + attenuation * color(scattered, world, depth + 1, do_emissive);
        } else {
            return emitted;
        }

    } else {

        //
        // fake sky-ish gradient
        //
        //vec3 unit_direction = normalize(r.direction());
        //float t = (unit_direction.y + 1.f) * .5;
        //return (1.f - t) * vec3{1.f, 1.f, 1.f} + t * vec3{.5f, .7f, 1.f};

        return {.0f, .0f, .0f};

    }
}

void progbar(size_t total, size_t samples, size_t* value, bool* quit)
{
    while (!(*quit))
    {
        float progress = float(*value / samples) / float(total);
        const size_t progbarsize = 70;

        std::cout << "tracing... [";
        for (size_t p = 0; p < progbarsize; ++p) {
            std::cout << ((p <= progress * progbarsize)? "#" : " ");
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << progress * 100.0f << "%\r";

        std::this_thread::sleep_for(10ms);
    }
}

int main(int argc, char** argv)
{
    const int nx = 512; // w
    const int ny = 512; // h
    const int ns = 100; // samples
    constexpr float inv_ns = 1.f / (float)ns;

    camera cam;

    //hitable* world = load_scene(eRANDOM, cam, float(nx) / float(ny));
    //hitable* world = load_scene(eCORNELLBOX, cam, float(nx) / float(ny));
    //hitable* world = load_scene(eFINAL, cam, float(nx) / float(ny));
    //hitable* world = load_scene(eTEST, cam, float(nx) / float(ny));
    hitable* world = load_scene(eFIRST_SCENE, cam, float(nx) / float(ny));

    char filename[256] = { "output.ppm" };
    if (argc == 2) {
        memset(filename, 0, 256);
        strncpy(filename, argv[1], strlen(argv[1]));
    }

    std::ofstream ppm_stream(filename, std::ios::binary);
    if (!ppm_stream.good()) {
        std::cerr << "unable to open " << filename << " for writing, abort\n";
        return -1;
    }

#if defined(TEST_PRNG)
    ppm_stream << "P6\n" << nx << " " << ny << " " << 0xff << "\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            uint8_t pixel = fastrand() * 255.99f;
            ppm_stream << pixel << pixel << pixel;
        }
    }
    ppm_stream.close();
    return 0;
#endif

    std::vector<vec3> output(nx * ny);

    // update progress bar using a separate thread
    bool quit = false;
    size_t pixel_idx = 0;
    std::thread progbar_thread(progbar, nx * ny, ns, &pixel_idx, &quit);

    // trace!
    Timer t;
    t.begin();

    //
    // OpenMP: collapse all 3 loops and distribute work to threads.
    //         scheduling must be dynamic to avoid work imbalance
    //         since rays could hit nothing or bounce "forever"
    //
#if !defined(_MSC_VER)
    #pragma omp parallel for collapse(3) schedule(dynamic)
#else
    //
    // ah, microsoft...
    //
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int j = 0; j < ny; ++j) {

        for (int i = 0; i < nx; ++i) {

            for (int s = 0; s < ns; ++s) {

                vec2 uv{ (i + fastrand()) / float(nx), (j + fastrand()) / float(ny) };
                Ray r = std::move(cam.get_ray(uv.x, uv.y));
                vec3 sampled_col = std::move(color(r, world, 0, true));

                #pragma omp atomic
                output[ny * j + i].r += sampled_col.r;

                #pragma omp atomic
                output[ny * j + i].g += sampled_col.g;

                #pragma omp atomic
                output[ny * j + i].b += sampled_col.b;

                pixel_idx++;
            }

        }

    }

    t.end();

    quit = true;
    progbar_thread.join();

    //
    // output to ppm (y inverted)
    //
    ppm_stream << "P6\n" << nx << " " << ny << " " << 0xff << "\n";
    for (int j = ny - 1; j >= 0; --j) {
        for (int i = 0; i < nx; ++i) {
            vec3 clamped_col = { clamp(255.99f * fastsqrt(output[ny * j + i].r * inv_ns), 0.0f, 255.0f),
                                 clamp(255.99f * fastsqrt(output[ny * j + i].g * inv_ns), 0.0f, 255.0f),
                                 clamp(255.99f * fastsqrt(output[ny * j + i].b * inv_ns), 0.0f, 255.0f) };

            ppm_stream << uint8_t(clamped_col.r) << uint8_t(clamped_col.g) << uint8_t(clamped_col.b);
        }
    }
    ppm_stream.close();

    std::cerr << "\nfinished in " << std::fixed << std::setprecision(1) << t.duration() << " secs\n";
}
