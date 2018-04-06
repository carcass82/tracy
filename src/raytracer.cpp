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
#include <cmath>
#include <cfloat>
#include <cassert>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb_image.h"

#include "timer.hpp"
#include "material.hpp"
#include "camera.hpp"
#include "hitable.hpp"
#include "ray.hpp"
#include "geom.hpp"
#include "scenes.hpp"

glm::vec3 color(const ray& r, hitable* world, int depth)
{
    hit_record rec;
    if (world->hit(r, 0.001f, FLT_MAX, rec)) {

        ray scattered;
        glm::vec3 attenuation;
        glm::vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

        if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
            return emitted + attenuation * color(scattered, world, depth + 1);
        } else {
            return emitted;
        }

    } else {
        return glm::vec3();
    }
}

int main(int argc, char** argv)
{
    const int nx = 512; // w
    const int ny = 512; // h
    const int ns = 100; // samples
    constexpr float inv_ns = 1.f / (float)ns;

    camera cam;

    hitable* world = load_scene(eRANDOM, cam, float(nx) / float(ny));
    //hitable* world = load_scene(eCORNELLBOX, cam, float(nx) / float(ny));
    //hitable* world = load_scene(eFINAL, cam, float(nx) / float(ny));
    //hitable* world = load_scene(eTEST, cam, float(nx) / float(ny));

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

    ppm_stream << "P6\n" << nx << " " << ny << " " << 0xff << "\n";

#if defined(TEST_PRNG)
    for (int j = ny - 1; j >= 0; --j) {

        for (int i = 0; i < nx; ++i) {

            uint8_t pixel = fastrand() * 255.99f;
            ppm_stream << pixel << pixel << pixel;

        }
    }
    ppm_stream.close();
    return 0;
#endif

    Timer t;
    t.begin();

    // path tracing
    size_t pixel_idx = 0;
    for (int j = ny - 1; j >= 0; --j) {

        for (int i = 0; i < nx; ++i) {

            glm::vec3 col;

            #pragma omp parallel for
            for (int s = 0; s < ns; ++s) {

                float u = float(i + fastrand()) / float(nx);
                float v = float(j + fastrand()) / float(ny);

                ray r = cam.get_ray(u, v);
                glm::vec3 temp = color(r, world, 0);

                #pragma omp atomic
                col.r += temp.r;

                #pragma omp atomic
                col.g += temp.g;

                #pragma omp atomic
                col.b += temp.b;
            }

            glm::vec3 clamped_col = glm::clamp(255.99f * col * inv_ns, 0.0f, 255.0f);

            float progress = float(pixel_idx++) / float(nx * ny);
            const size_t progbarsize = 70;

            std::cout << "tracing... [";
            for (size_t p = 0; p < progbarsize; ++p) {
                std::cout << ((p <= progress * progbarsize)? "=" : " ");
            }
            std::cout << "] " << std::fixed << std::setprecision(1) << progress * 100.0f << "%\r";

            ppm_stream << uint8_t(clamped_col.r) << uint8_t(clamped_col.g) << uint8_t(clamped_col.b);
        }

    }

    ppm_stream.close();

    t.end();
    std::cerr << "\nfinished in " << std::fixed << std::setprecision(3) << t.duration() << " secs\n";
}
