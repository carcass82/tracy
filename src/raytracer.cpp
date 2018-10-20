/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 *
 *
 * TODO:
 * - read scene desc from file
 * - support for triangular meshes
 * - openmp -> gpu
 * - realtime preview of result
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
using cc::math::vec2;
using cc::math::vec3;
using cc::util::clamp;

#if defined(_MSC_VER)
 #define MAINCALLCONV __cdecl
 #define NOVTABLE __declspec(novtable)
#else
 #define MAINCALLCONV
 #define NOVTABLE
#endif

#if defined(USE_CUDA)
extern "C" void MAINCALLCONV cuda_trace(int, int, int, float*, size_t&);
#endif

#include "timer.hpp"
#include "material.hpp"
#include "camera.hpp"
#include "hitable.hpp"
#include "ray.hpp"
#include "geom.hpp"
#include "pdf.hpp"
#include "pdf/hitable.hpp"
#include "pdf/cosine.hpp"
#include "pdf/mix.hpp"
#include "scenes.hpp"

// max "bounces" for tracing
#define MAX_DEPTH 30

class custom_pdf : public pdf
{
public:
    float value(const vec3& direction) const override final { return .5f; }
    vec3 generate() const override final { return vec3{1, 0, 0}; }

    //
    // avoid memleaks or simply out-of-memory for too many new PDFs
    // (at the cost of objects created and destoyed each loop - profiler does seem happy though!)
    //
    static void generate_all(hitable* shape, const vec3& o, const vec3& w, Ray& out_scattered, float& out_pdf)
    {
        if (shape)
        {
            hitable_pdf objectpdf(shape, o);
            cosine_pdf cospdf(w);

            mix_pdf mixpdf(&cospdf, &objectpdf, .45f);

            out_scattered = Ray(o, mixpdf.generate());
            out_pdf = mixpdf.value(out_scattered.GetDirection());
        }
        else
        {
            cosine_pdf cospdf(w);
            out_scattered = Ray(o, cospdf.generate());
            out_pdf = cospdf.value(out_scattered.GetDirection());
        }
    }
};


vec3 color(const Ray& r, hitable* world, int depth, hitable* light_shape, size_t& raycount)
{
    ++raycount;

    hit_record rec;
    if (world->hit(r, 0.01f, std::numeric_limits<float>::max(), rec))
    {
        scatter_record srec;
        vec3 emitted = rec.mat_ptr->emitted(r, rec, rec.uv, rec.p);

        if (depth < MAX_DEPTH && rec.mat_ptr->scatter(r, rec, srec))
        {
            if (srec.is_specular)
            {
                return srec.attenuation * color(srec.specular, world, depth + 1, light_shape, raycount);
            }

            Ray scattered;
            float pdf_val;
            custom_pdf::generate_all(light_shape, rec.p, rec.normal, scattered, pdf_val);

            return emitted + srec.attenuation * rec.mat_ptr->scattering_pdf(r, rec, scattered) * color(scattered, world, depth + 1, light_shape, raycount) / pdf_val;
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
        //vec3 unit_direction = normalize(r.GetDirection());
        //float t = (unit_direction.y + 1.f) * .5;
        //return (1.f - t) * vec3{1.f, 1.f, 1.f} + t * vec3{.5f, .7f, 1.f};

        return vec3{ .0f, .0f, .0f };
    }
}

#if !defined(USE_CUDA)
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

int MAINCALLCONV main(int argc, char** argv)
{
    const int nx = 800; // w
    const int ny = 600; // h
    const int ns = 250; // samples

    camera cam;

    // test
    //hitable* world = load_scene(eRANDOM, cam, float(nx) / float(ny));
    //hitable* list[] =
    //{
    //    new xz_rect(-20, 20, -20, 20, 10, nullptr) // light
    //};
    //hitable_list* hlist = new hitable_list(list, cc::util::array_size(list));
    
    // modified cornell box
    //hitable* world = load_scene(eCORNELLBOX, cam, float(nx) / float(ny));
    hitable* world = load_scene(eTESTGPU, cam, float(nx) / float(ny));
    hitable* list[] =
    {
        new xz_rect(213, 343, 227, 332, 554, nullptr)
        /*, // light
        new sphere(vec3(130 + 82.5 - 25, 215, 65 + 82.5 - 25), 50.0, nullptr) // glass sphere*/
    };
    hitable_list* hlist = new hitable_list(list, cc::util::array_size(list));

#if !defined(USE_CUDA)
    char filename[256] = { "output.ppm" };
#else
    char filename[256] = { "output_cuda.ppm" };
#endif
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

    // output buffer
    vec3* output = new vec3[nx * ny];

    // trace!
    Timer t;
    t.begin();

    size_t totrays = 0;

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
                Ray r = cam.get_ray(uv.x, uv.y);
                vec3 sampled_col = color(r, world, 0, hlist, raycount);

                #pragma omp atomic
                output[j * nx + i].r += sampled_col.r;

                #pragma omp atomic
                output[j * nx + i].g += sampled_col.g;

                #pragma omp atomic
                output[j * nx + i].b += sampled_col.b;

                // not really interested in correctness
                pixel_idx++;

                //#pragma omp atomic
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

    std::cout << "Average: " << std::fixed << std::setprecision(2) << (totrays / 1000000.0) / t.duration() << " MRays/s\n";

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

    std::cerr << "finished in " << std::fixed << std::setprecision(1) << t.duration() << " secs\n" << std::endl;
}
