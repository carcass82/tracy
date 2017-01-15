/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include <vector>
#include <algorithm>
#include <cmath>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

glm::vec3 random_in_unit_sphere()
{
    glm::vec3 p;
    do
        p = 2.0f * glm::vec3(drand48(), drand48(), drand48()) - glm::vec3(1.0f, 1.0f, 1.0f);
    while (length2(p) >= 1.0f);

    return p;
}

glm::vec3 random_in_unit_disk()
{
    glm::vec3 p;
    do
        p = 2.0f * glm::vec3(drand48(), drand48(), 0.0f) - glm::vec3(1.0f, 1.0f, 0.0f);
    while (length2(p) >= 1.0f);

    return p;
}

float schlick(float cos, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 *= r0;
    
    return r0 + (1.0f - r0) * glm::pow((1 - cos), 5);
}

void histogram_equalize(std::vector<float>& values)
{
    size_t total = values.size();
    size_t n_bins = 256;
    
    std::vector<uint16_t> hist(n_bins, 0);
    for (size_t i = 0; i < total; ++i)
        hist[uint16_t(values[i] * 255.99)]++;
        
    size_t i = 0;
    while (hist[i] == 0)
        ++i;
        
    if (hist[i] == total) {
        for (size_t j = 0; j < total; ++j) {
            values[j] = i / 255.0f;
        }
        
        return;
    }
    
    float scale = (n_bins - 1.0f) / (total - hist[i]);
    
    std::vector<uint16_t> lut(n_bins, 0);
    ++i;
    
    uint32_t sum = 0;
    for (; i < hist.size(); ++i) {
        sum += hist[i];
        lut[i] = glm::clamp(uint32_t(glm::round(sum * scale)), 0u, 255u);
    }
    
    for (size_t i = 0; i < total; ++i) {
        values[i] = lut[uint16_t(values[i] * 255.99)] / 255.0f;
    }
}
