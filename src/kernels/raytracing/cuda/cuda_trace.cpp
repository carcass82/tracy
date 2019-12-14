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
extern "C" void cuda_trace(CUDAScene* scene, unsigned* out, int w, int h);


struct CUDATrace::CUDATraceDetails
{
    CUDAScene scene_;

    GLuint vs;
    GLuint fs;
    GLuint shader;
    GLuint texture;
    cudaGraphicsResource* mapped_texture;
    unsigned int* output_buffer;
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
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, win_width_, win_height_, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, NULL);

        CUDAAssert(cudaGraphicsGLRegisterImage(&details_->mapped_texture, details_->texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

        static const char* vertex_shader = R"vs(
        void main()
        {
            gl_Position = gl_Vertex;
            gl_TexCoord[0].xy = gl_MultiTexCoord0.xy;
        }
        )vs";

        static const char* fragment_shader = R"fs(
        #version 130

        uniform usampler2D in_texture;
        void main()
        {
            vec4 ucolor = texture(in_texture, gl_TexCoord[0].xy);
            gl_FragColor = ucolor / 255.0;
        }
        )fs";

        // Create Shaders
        {
            details_->vs = glCreateShader(GL_VERTEX_SHADER);
            glShaderSource(details_->vs, 1, &vertex_shader, NULL);
            glCompileShader(details_->vs);

            details_->fs = glCreateShader(GL_FRAGMENT_SHADER);
            glShaderSource(details_->fs, 1, &fragment_shader, NULL);
            glCompileShader(details_->fs);

#if defined(_DEBUG)
            static char buffer[512];
            GLint status;

            glGetShaderiv(details_->vs, GL_COMPILE_STATUS, &status);
            if (status != GL_TRUE)
            {
                glGetShaderInfoLog(details_->vs, array_size(buffer), nullptr, buffer);
                __debugbreak();
            }

            glGetShaderiv(details_->fs, GL_COMPILE_STATUS, &status);
            if (status != GL_TRUE)
            {
                glGetShaderInfoLog(details_->fs, array_size(buffer), nullptr, buffer);
                __debugbreak();
            }
#endif

            details_->shader = glCreateProgram();
            glAttachShader(details_->shader, details_->vs);
            glAttachShader(details_->shader, details_->fs);
            glLinkProgram(details_->shader);

#if defined(_DEBUG)
            glGetProgramiv(details_->shader, GL_LINK_STATUS, &status);
            if (status != GL_TRUE)
            {
                glGetProgramInfoLog(details_->shader, array_size(buffer), nullptr, buffer);
                __debugbreak();
            }
#endif
        }

        CUDAAssert(cudaMalloc((void**)&details_->output_buffer, win_width_ * win_height_ * sizeof(GLuint)));
    }

    cuda_setup(in_scene, &details_->scene_);
}

void CUDATrace::UpdateScene()
{
    cuda_trace(&details_->scene_, details_->output_buffer, win_width_, win_height_);

    cudaArray* texture_ptr;
    CUDAAssert(cudaGraphicsMapResources(1, &details_->mapped_texture, 0));
    CUDAAssert(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, details_->mapped_texture, 0, 0));
    CUDAAssert(cudaMemcpyToArray(texture_ptr, 0, 0, details_->output_buffer, win_width_ * win_height_ * sizeof(GLuint), cudaMemcpyDeviceToDevice));
    CUDAAssert(cudaGraphicsUnmapResources(1, &details_->mapped_texture, 0));
}

void CUDATrace::RenderScene()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(details_->shader);
    glUniform1i(glGetUniformLocation(details_->shader, "in_texture"), 0);

    glBindTexture(GL_TEXTURE_2D, details_->texture);

    glBegin(GL_QUADS);
     glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
     glTexCoord2f(1.0, 0.0); glVertex2f( 1.0, -1.0);
     glTexCoord2f(1.0, 1.0); glVertex2f( 1.0,  1.0);
     glTexCoord2f(0.0, 1.0); glVertex2f(-1.0,  1.0);
    glEnd();

    glUseProgram(0);

#if defined(WIN32)
    SwapBuffers(GetDC(win_handle_));
#else
    glXSwapBuffers(win_handle_->dpy, win_handle_->win);
#endif
}
