/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <cfloat>
#include <cassert>
#include <thread>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb_image.h"

#include "ext/cclib/cclib.h"

#include "timer.hpp"
#include "material.hpp"
#include "camera.hpp"
#include "hitable.hpp"
#include "ray.hpp"
#include "geom.hpp"
#include "pdf.hpp"
#include "scenes.hpp"

using cc::math::vec2;
using cc::math::vec3;
using cc::util::clamp;

#if defined(_MSC_VER)
 #define MAINCALLCONV __cdecl
#else
 #define MAINCALLCONV
#endif

vec3 color(const Ray& r, hitable* world, int depth, hitable* light_shape)
{
    hit_record rec;
    if (world->hit(r, 0.01f, std::numeric_limits<float>::max(), rec))
    {
        scatter_record srec;
        vec3 emitted = rec.mat_ptr->emitted(r, rec, rec.uv, rec.p);

        if (depth < 50 && rec.mat_ptr->scatter(r, rec, srec))
        {
            if (srec.is_specular)
            {
                return srec.attenuation * color(srec.specular, world, depth + 1, light_shape);
            }

            Ray scattered;
            float pdf_val;
            custom_pdf::generate_all(light_shape, rec.p, rec.normal, scattered, pdf_val);

            return emitted + srec.attenuation * rec.mat_ptr->scattering_pdf(r, rec, scattered) * color(scattered, world, depth + 1, light_shape) / pdf_val;
        }
        else
        {
            return emitted;
        }

    }
    else
    {
        //
        // fake sky-ish gradient
        //
        //vec3 unit_direction = normalize(r.direction());
        //float t = (unit_direction.y + 1.f) * .5;
        //return (1.f - t) * vec3{1.f, 1.f, 1.f} + t * vec3{.5f, .7f, 1.f};

        return vec3{ .0f, .0f, .0f };
    }
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

        std::this_thread::sleep_for(16ms);
    }

    std::cout << "tracing... [" << std::string(progbarsize, '#') << "] 100.0%\n";
}

int MAINCALLCONV main(int argc, char** argv)
{
    const int nx = 512; // w
    const int ny = 512; // h
    const int ns = 200; // samples

    camera cam;

    //hitable* world = load_scene(eRANDOM, cam, float(nx) / float(ny));

    hitable* world = load_scene(eCORNELLBOX, cam, float(nx) / float(ny));
    hitable* list[] =
    {
        new xz_rect(213, 343, 227, 332, 554, nullptr), // light
        //new sphere(vec3(190, 90, 190), 90.0, nullptr)  // glass sphere
    };
    hitable_list* hlist = new hitable_list(list, cc::util::array_size(list));

    //hitable* world = load_scene(eFINAL, cam, float(nx) / float(ny));
    //hitable* world = load_scene(eTEST, cam, float(nx) / float(ny));

    //
    // first book, scene with 3 different big spheres
    //
    //hitable* world = load_scene(eFIRST_SCENE, cam, float(nx) / float(ny));
    //hitable* list[] =
    //{
    //    new xz_rect(-8, 8, -8, 8, 10, nullptr),
    //    new sphere(vec3(-4.f, 1.f, .0f), 1.f, nullptr)
    //};
    //hitable_list* hlist = new hitable_list(list, vutil::array_size(list));

    char filename[256] = { "output.ppm" };
    if (argc == 2)
    {
        memset(filename, 0, 256);
        strncpy(filename, argv[1], 255);
    }

    std::ofstream ppm_stream(filename, std::ios::binary);
    if (!ppm_stream.good())
    {
        std::cerr << "unable to open " << filename << " for writing, abort\n";
        return -1;
    }

#if defined(TEST_PRNG)
    ppm_stream << "P6\n" << nx << " " << ny << " " << 0xff << "\n";
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
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
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            for (int s = 0; s < ns; ++s)
            {
                vec2 uv{ (i + fastrand()) / float(nx), (j + fastrand()) / float(ny) };
                Ray r = cam.get_ray(uv.x, uv.y);
                vec3 sampled_col = color(r, world, 0, hlist);

                #pragma omp atomic
                output[ny * j + i].r += sampled_col.r;

                #pragma omp atomic
                output[ny * j + i].g += sampled_col.g;

                #pragma omp atomic
                output[ny * j + i].b += sampled_col.b;

                // not really interested in correctness
                pixel_idx++;
            }
        }
    }
    t.end();

    quit = true;
    progbar_thread.join();

    //
    // output to ppm (y inverted)
    // gamma 2.0
    //
    ppm_stream << "P6\n" << nx << " " << ny << " " << 0xff << "\n";
    for (int j = ny - 1; j >= 0; --j)
    {
        for (int i = 0; i < nx; ++i)
        {
            vec3 clamped_col = vec3{ clamp(255.99f * sqrtf(output[ny * j + i].r / ns), 0.0f, 255.0f),
                                     clamp(255.99f * sqrtf(output[ny * j + i].g / ns), 0.0f, 255.0f),
                                     clamp(255.99f * sqrtf(output[ny * j + i].b / ns), 0.0f, 255.0f) };

            ppm_stream << uint8_t(clamped_col.r) << uint8_t(clamped_col.g) << uint8_t(clamped_col.b);
        }
    }
    ppm_stream.close();

    std::cerr << "finished in " << std::fixed << std::setprecision(1) << t.duration() << " secs\n";
}
