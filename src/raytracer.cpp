/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
 
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cfloat>

#if defined(_WIN32)
double drand48() { return (rand() / (RAND_MAX + 1.0)); }
#endif

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

        // debug - blueish gradient
        //vec3 unit_direction = glm::normalize(r.direction());
        //float t = 0.5f * (unit_direction.y + 1.0f);
        //return (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);

        // debug - white "ambient" light
        //return glm::vec3(0.01,0.01,0.01);
    }
}

int main()
{
    const int nx = 500; // w
    const int ny = 500; // h
    const int ns = 100; // samples

    camera cam;
    
    //hitable* world = load_scene(eRANDOM, cam, float(nx) / float(ny));
    hitable* world = load_scene(eCORNELLBOX, cam, float(nx) / float(ny));
    //hitable* world = load_scene(eFINAL, cam, float(nx) / float(ny));
    //hitable* world = load_scene(eTEST, cam, float(nx) / float(ny));
    
    Timer t;
    t.begin();

    // PPM file header
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";

    std::vector<glm::vec3> output;
    output.reserve(nx * ny);

    // path tracing
    for (int j = ny - 1; j >= 0; --j) {

        for (int i = 0; i < nx; ++i) {

            glm::vec3 col;

            #pragma omp parallel for
            for (int s = 0; s < ns; ++s) {

                float u = float(i + drand48()) / float(nx);
                float v = float(j + drand48()) / float(ny);

                ray r = cam.get_ray(u, v);
                glm::vec3 tempcol = color(r, world, 0);
                
                #pragma omp atomic
                col[0] += tempcol[0];
                
                #pragma omp atomic
                col[1] += tempcol[1];
                
                #pragma omp atomic
                col[2] += tempcol[2];
            }
            
            col /= float(ns);

            // gamma correct 2.0
            col = glm::sqrt(col);

            // TODO: should not be clamped but histogram equalized
            //cout << int(glm::clamp(255.99f * col[0], 0.0f, 255.0f)) << " "
            //     << int(glm::clamp(255.99f * col[1], 0.0f, 255.0f)) << " "
            //     << int(glm::clamp(255.99f * col[2], 0.0f, 255.0f)) << "\n";
            
            output.push_back(col);
        }

    }

    t.end();
    std::cerr << "finished in " << std::setprecision(3) << t.duration() << " secs\n";

    const glm::vec3 yuv(0.299f, 0.114f, 0.587f);
    std::vector<float> y_vals;
    std::vector<float> u_vals;
    std::vector<float> v_vals;
    for (auto pixel : output) {
        float y = pixel.r * yuv.r + pixel.g * yuv.g + pixel.b * yuv.b;
        
        u_vals.push_back(0.436f * (pixel.b - y) / (1.0f - yuv.b));
        v_vals.push_back(0.615f * (pixel.r - y) / (1.0f - yuv.r));
        y_vals.push_back(y);
    }
    
    //histogram_equalize(y_vals);
    
    for (size_t i = 0; i < y_vals.size(); ++i) {
        
        float r = y_vals[i] + v_vals[i] * ((1.0f - yuv.r) / 0.615f);
        float g = y_vals[i] - (u_vals[i] * (yuv.b * (1.0f - yuv.b)) / (0.436f * yuv.g)) - (v_vals[i] * (yuv.r * (1.0f - yuv.r)) / (0.615f * yuv.g));
        float b = y_vals[i] + u_vals[i] * ((1.0f - yuv.b) / 0.436f);
        
        std::cout << int(255.99f * r) << " " << int(255.99f * g) << " " << int(255.99f * b) << "\n";
    }
    
    std::cout << std::endl;
}
