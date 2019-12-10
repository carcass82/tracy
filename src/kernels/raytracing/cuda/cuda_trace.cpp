/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "cuda_trace.h"

struct CUDATrace::CUDATraceDetails
{
};

CUDATrace::CUDATrace()
    : details_(new CUDATraceDetails)
{
}

CUDATrace::~CUDATrace()
{
    delete details_;
}

void CUDATrace::Initialize(Handle in_window, int in_width, int in_height, const Scene& in_scene)
{
    win_handle_ = in_window;
    win_width_ = in_width;
    win_height_ = in_height;
    camera_ = &in_scene.GetCamera();
    scene_ = &in_scene;
}
