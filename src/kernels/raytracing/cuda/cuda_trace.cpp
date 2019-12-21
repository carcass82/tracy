/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "cuda_trace.h"
#include "GL/glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "log.h"
#include "cuda_mesh.h"
#include "cuda_scene.h"

#if !defined(WIN32)
#include <GL/glx.h>
#endif

extern "C" void cuda_setup(const Scene & in_scene, CUDAScene* out_scene);
extern "C" void cuda_trace(CUDAScene* scene, int framecount);


struct CUDATrace::CUDATraceDetails
{
    CUDAScene scene_;

    GLuint vs;
    GLuint fs;
    GLuint shader;
    GLuint texture;
    cudaGraphicsResource* mapped_texture;
};

CUDATrace::CUDATrace()
    : details_(new CUDATraceDetails)
{
}

CUDATrace::~CUDATrace()
{
    delete details_;
}

int CUDATrace::GetRayCount() const
{
    CUDAAssert(cudaMemcpy(&details_->scene_.h_raycount, details_->scene_.d_raycount, sizeof(int), cudaMemcpyDeviceToHost));
    return details_->scene_.h_raycount;
}

void CUDATrace::ResetRayCount()
{
    CUDAAssert(cudaMemset(details_->scene_.d_raycount, 0, sizeof(int)));
    details_->scene_.h_raycount = 0;
}

void CUDATrace::Initialize(Handle in_window, int in_width, int in_height, const Scene& in_scene)
{
    win_handle_ = in_window;
    win_width_ = in_width;
    win_height_ = in_height;

#if defined(WIN32)
    PIXELFORMATDESCRIPTOR pfd;
    memset(&pfd, 0, sizeof(PIXELFORMATDESCRIPTOR));
    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    pfd.cDepthBits = 16;
    pfd.iLayerType = PFD_MAIN_PLANE;

    HDC hDC = GetDC(win_handle_);
    GLuint PixelFormat = ChoosePixelFormat(hDC, &pfd);
    SetPixelFormat(hDC, PixelFormat, &pfd);
    HGLRC hRC = wglCreateContext(hDC);
    wglMakeCurrent(hDC, hRC);
#else
    GLint att[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };
    XVisualInfo* vi = glXChooseVisual(win_handle_->dpy, 0, att);
    GLXContext glc = glXCreateContext(win_handle_->dpy, vi, nullptr, GL_TRUE);
    glXMakeCurrent(win_handle_->dpy, win_handle_->win, glc);
#endif

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err == GLEW_OK)
    {
        glEnable(GL_FRAMEBUFFER_SRGB);

        glDisable(GL_DEPTH_TEST);
        
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0, 0, win_width_, win_height_);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glGenTextures(1, &details_->texture);
        glBindTexture(GL_TEXTURE_2D, details_->texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, win_width_, win_height_, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        CUDAAssert(cudaSetDevice(CUDA_PREFERRED_DEVICE));
        
        CUDAAssert(cudaGraphicsGLRegisterImage(&details_->mapped_texture, details_->texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
        CUDAAssert(cudaMalloc(&details_->scene_.d_output_, win_width_ * win_height_ * sizeof(vec4)));
        CUDAAssert(cudaMemset(details_->scene_.d_output_, 0, win_width_ * win_height_ * sizeof(vec4)));
    }

    details_->scene_.width = win_width_;
    details_->scene_.height = win_height_;
    cuda_setup(in_scene, &details_->scene_);
}

void CUDATrace::UpdateScene()
{
    cuda_trace(&details_->scene_, frame_counter_++);

    cudaArray* texture_ptr;
    CUDAAssert(cudaGraphicsMapResources(1, &details_->mapped_texture, 0));
    CUDAAssert(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, details_->mapped_texture, 0, 0));
    CUDAAssert(cudaMemcpyToArray(texture_ptr, 0, 0, details_->scene_.d_output_, win_width_ * win_height_ * sizeof(vec4), cudaMemcpyDeviceToDevice));
    CUDAAssert(cudaGraphicsUnmapResources(1, &details_->mapped_texture, 0));
}

void CUDATrace::RenderScene()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, details_->texture);
    glEnable(GL_TEXTURE_2D);

    glBegin(GL_QUADS);
     glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
     glTexCoord2f(1.0, 0.0); glVertex2f( 1.0, -1.0);
     glTexCoord2f(1.0, 1.0); glVertex2f( 1.0,  1.0);
     glTexCoord2f(0.0, 1.0); glVertex2f(-1.0,  1.0);
    glEnd();

#if defined(WIN32)
    SwapBuffers(GetDC(win_handle_));
#else
    glXSwapBuffers(win_handle_->dpy, win_handle_->win);
#endif
}
