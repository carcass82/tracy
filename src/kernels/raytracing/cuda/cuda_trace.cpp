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
#include "cuda_scene.h"
#include "collision.h"

#if !defined(WIN32)
#include <GL/glx.h>
#endif

extern "C" void cuda_setup(const Scene& in_scene, CUDAScene* out_scene);
extern "C" void cuda_trace(CUDAScene* scene, int framecount);
extern "C" void cuda_shutdown(CUDAScene* out_scene);


struct CUDATrace::CUDATraceDetails
{
    CUDAScene scene_;

    GLuint texture;
    cudaGraphicsResource* mapped_texture;
    cudaArray* texture_content;

    void PrepareSceneTree(const Scene& in_scene)
    {
        // fill a tree with just one node containing all elements in the scene
        accel::Node<TriInfo> SceneTree;
        SceneTree.GetElements().reserve(in_scene.GetTriCount());
        {
            BBox scene_bbox{ FLT_MAX, -FLT_MAX };
            for (unsigned int i = 0; i < in_scene.GetObjectCount(); ++i)
            {
                const Mesh& mesh = in_scene.GetObject(i);
                if (mesh.GetTriCount() * 3 > pow(2, 24) - 1)
                {
                    TracyLog("Unable to represent triangle index\n");
                    DEBUG_BREAK();
                }

                for (unsigned int t = 0; t < mesh.GetTriCount(); ++t)
                {
                    SceneTree.GetElements().emplace_back(i, t);
                }

                scene_bbox.minbound = pmin(mesh.GetAABB().minbound, scene_bbox.minbound);
                scene_bbox.maxbound = pmax(mesh.GetAABB().maxbound, scene_bbox.maxbound);
            }
            SceneTree.SetAABB(scene_bbox);
        }

        // process tree
        accel::BuildTree<TriInfo>(&SceneTree, [&in_scene](const auto& in_triangle, const BBox& in_aabb)
            {
                // Triangle-AABB intersection test for tree building
                uint32_t mesh_id = in_triangle.GetMeshId();
                uint32_t triangle_id = in_triangle.GetTriangleId() * 3;
                const Mesh& mesh = in_scene.GetObject(mesh_id);
                vec3 v0{ mesh.GetVertex(mesh.GetIndex(triangle_id + 0)).pos - in_aabb.GetCenter() };
                vec3 v1{ mesh.GetVertex(mesh.GetIndex(triangle_id + 1)).pos - in_aabb.GetCenter() };
                vec3 v2{ mesh.GetVertex(mesh.GetIndex(triangle_id + 2)).pos - in_aabb.GetCenter() };

                return collision::TriangleAABB(v0, v1, v2, in_aabb);
            }
        );

        // flatten tree to ease upload on device
        accel::FlattenTree<TriInfo>(SceneTree, scene_.h_scenetree);
    }
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
    return details_->scene_.GetRayCount();
}

void CUDATrace::ResetRayCount()
{
    details_->scene_.ResetRayCount();
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

        CUDAAssert(cudaSetDevice(0));
        CUDAAssert(cudaGraphicsGLRegisterImage(&details_->mapped_texture, details_->texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
    }

    camera_ = &in_scene.GetCamera();

    details_->scene_.width = win_width_;
    details_->scene_.height = win_height_;

#if USE_KDTREE
    details_->PrepareSceneTree(in_scene);
#endif

    cuda_setup(in_scene, &details_->scene_);
}

void CUDATrace::UpdateScene()
{
    if (camera_->IsDirty())
    {
        CUDAAssert(cudaMemcpy(details_->scene_.d_camera_, camera_, sizeof(Camera), cudaMemcpyHostToDevice));
    }

    int frame_counter = (int)camera_->BeginFrame();
    cuda_trace(&details_->scene_, frame_counter);
    camera_->EndFrame();

    CUDAAssert(cudaSetDevice(0));
    CUDAAssert(cudaGraphicsMapResources(1, &details_->mapped_texture, 0));
    CUDAAssert(cudaGraphicsSubResourceGetMappedArray(&details_->texture_content, details_->mapped_texture, 0, 0));
    CUDAAssert(cudaMemcpy2DToArray(details_->texture_content, 0, 0, details_->scene_.d_output_, win_width_ * sizeof(vec4), win_width_ * sizeof(vec4), win_height_, cudaMemcpyDeviceToDevice));
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

void CUDATrace::Shutdown()
{
    cuda_shutdown(&details_->scene_);
}
