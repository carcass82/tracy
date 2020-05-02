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

extern "C" void cuda_setup(const Scene& in_scene, CUDAScene* out_scene);
extern "C" void cuda_trace(CUDAScene* scene, int framecount);


struct CUDATrace::CUDATraceDetails
{
    CUDAScene scene_;

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

        CUDAAssert(cudaSetDevice(CUDA_PREFERRED_DEVICE));
        
        CUDAAssert(cudaGraphicsGLRegisterImage(&details_->mapped_texture, details_->texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
        CUDAAssert(cudaMalloc(&details_->scene_.d_output_, win_width_ * win_height_ * sizeof(vec4)));
        CUDAAssert(cudaMemset(details_->scene_.d_output_, 0, win_width_ * win_height_ * sizeof(vec4)));
    }

    camera_ = &in_scene.GetCamera();

    details_->scene_.width = win_width_;
    details_->scene_.height = win_height_;

#if USE_KDTREE

    // pretend it's a function like PrepareScene()
    {
		// Triangle-AABB intersection
		auto TriangleAABBTester = [](const Triangle& triangle, const BBox& aabb)
		{
			// triangle - box test using separating axis theorem (https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/pubs/tribox.pdf)
			// code adapted from http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt

			vec3 v0{ triangle.v[0] - aabb.GetCenter() };
			vec3 v1{ triangle.v[1] - aabb.GetCenter() };
			vec3 v2{ triangle.v[2] - aabb.GetCenter() };

			vec3 e0{ v1 - v0 };
			vec3 e1{ v2 - v1 };
			vec3 e2{ v0 - v2 };

			vec3 fe0{ abs(e0.x), abs(e0.y), abs(e0.z) };
			vec3 fe1{ abs(e1.x), abs(e1.y), abs(e1.z) };
			vec3 fe2{ abs(e2.x), abs(e2.y), abs(e2.z) };

			vec3 aabb_hsize = aabb.GetSize() / 2.f;

			auto AxisTester = [](float a, float b, float fa, float fb, float v0_0, float v0_1, float v1_0, float v1_1, float hsize_0, float hsize_1)
			{
				float p0 = a * v0_0 + b * v0_1;
				float p1 = a * v1_0 + b * v1_1;

				float rad = fa * hsize_0 + fb * hsize_1;
				return (min(p0, p1) > rad || max(p0, p1) < -rad);
			};

			if (AxisTester(e0.z, -e0.y, fe0.z, fe0.y, v0.y, v0.z, v2.y, v2.z, aabb_hsize.y, aabb_hsize.z) ||
				AxisTester(-e0.z, e0.x, fe0.z, fe0.x, v0.x, v0.z, v2.x, v2.z, aabb_hsize.x, aabb_hsize.z) ||
				AxisTester(e0.y, -e0.x, fe0.y, fe0.x, v1.x, v1.y, v2.x, v2.y, aabb_hsize.x, aabb_hsize.y) ||

				AxisTester(e1.z, -e1.y, fe1.z, fe1.y, v0.y, v0.z, v2.y, v2.z, aabb_hsize.y, aabb_hsize.z) ||
				AxisTester(-e1.z, e1.x, fe1.z, fe1.x, v0.x, v0.z, v2.x, v2.z, aabb_hsize.x, aabb_hsize.z) ||
				AxisTester(e1.y, -e1.x, fe1.y, fe1.x, v0.x, v0.y, v1.x, v1.y, aabb_hsize.x, aabb_hsize.y) ||

				AxisTester(e2.z, -e2.y, fe2.z, fe2.y, v0.y, v0.z, v1.y, v1.z, aabb_hsize.y, aabb_hsize.z) ||
				AxisTester(-e2.z, e2.x, fe2.z, fe2.x, v0.x, v0.z, v1.x, v1.z, aabb_hsize.x, aabb_hsize.z) ||
				AxisTester(e2.y, -e2.x, fe2.y, fe2.x, v1.x, v1.y, v2.x, v2.y, aabb_hsize.x, aabb_hsize.y))
			{
				return false;
			}

			vec3 trimin = pmin(v0, pmin(v1, v2));
			vec3 trimax = pmax(v0, pmax(v1, v2));
			if ((trimin.x > aabb_hsize.x || trimax.x < -aabb_hsize.x) ||
				(trimin.y > aabb_hsize.y || trimax.y < -aabb_hsize.y) ||
				(trimin.z > aabb_hsize.z || trimax.z < -aabb_hsize.z))
			{
				return false;
			}

			{
				vec3 trinormal = cross(e0, e1);

				vec3 vmin, vmax;

				if (trinormal.x > .0f) { vmin.x = -aabb_hsize.x - v0.x; vmax.x = aabb_hsize.x - v0.x; }
				else { vmin.x = aabb_hsize.x - v0.x; vmax.x = -aabb_hsize.x - v0.x; }

				if (trinormal.y > .0f) { vmin.y = -aabb_hsize.y - v0.y; vmax.y = aabb_hsize.y - v0.y; }
				else { vmin.y = aabb_hsize.y - v0.y; vmax.y = -aabb_hsize.y - v0.y; }

				if (trinormal.z > .0f) { vmin.z = -aabb_hsize.z - v0.z; vmax.z = aabb_hsize.z - v0.z; }
				else { vmin.z = aabb_hsize.z - v0.z; vmax.z = -aabb_hsize.z - v0.z; }

				if (dot(trinormal, vmin) > .0f || dot(trinormal, vmax) < .0f)
				{
					return false;
				}
			}

			return true;
		};

		if (in_scene.GetObjectCount() > UINT16_MAX)
		{
			TracyLog("Unable to represent mesh index\n");
			DEBUG_BREAK();
		}

		BBox scene_bbox{ FLT_MAX, -FLT_MAX };
		vector<Triangle> scene_tris;
		for (uint16_t i = 0; i < in_scene.GetObjectCount(); ++i)
		{
			const Mesh& mesh = in_scene.GetObject(i);
			if (mesh.GetTriCount() * 3 > UINT16_MAX)
			{
				TracyLog("Unable to represent triangle index\n");
				DEBUG_BREAK();
			}

			for (uint16_t t = 0; t < mesh.GetTriCount(); ++t)
			{
				uint16_t tri_idx = t * 3;

				const vec3& v0 = mesh.GetVertex(mesh.GetIndex(tri_idx + 0)).pos;
				const vec3& v1 = mesh.GetVertex(mesh.GetIndex(tri_idx + 1)).pos;
				const vec3& v2 = mesh.GetVertex(mesh.GetIndex(tri_idx + 2)).pos;

				scene_bbox.minbound = pmin(mesh.GetAABB().minbound, scene_bbox.minbound);
				scene_bbox.maxbound = pmax(mesh.GetAABB().maxbound, scene_bbox.maxbound);

				scene_tris.emplace_back(v0, v1, v2, i, tri_idx);
			}
		}

        accel::Node<Triangle> SceneTree;
		SceneTree.SetAABB(scene_bbox);
		SceneTree.GetElements().assign(scene_tris.begin(), scene_tris.end());
        accel::BuildTree<Triangle, std::vector>(&SceneTree, TriangleAABBTester);

		vector<CustomNode<CUDATree, Triangle>> nodes;
		vector<Triangle> triangles;
		{
			using TriangleNode = CustomNode<CUDATree, Triangle>;

			vector<std::pair<unsigned int, accel::Node<Triangle>*>> build_queue;
			nodes.push_back(TriangleNode());

			build_queue.push_back(std::pair(0, &SceneTree));
			while (!build_queue.empty())
			{
				auto current_node = build_queue.back();
				build_queue.pop_back();

				if (current_node.second)
				{
					nodes[current_node.first].aabb = current_node.second->GetAABB();

					if (current_node.second->IsEmpty())
					{
						unsigned int right_children = (unsigned int)nodes.size();
						nodes[current_node.first].children[Child::Right] = right_children;
						nodes.push_back(TriangleNode());
						build_queue.push_back(std::pair(right_children, current_node.second->GetChild(accel::Child::Right)));

						unsigned int left_children = (unsigned int)nodes.size();
						nodes[current_node.first].children[Child::Left] = left_children;
						nodes.push_back(TriangleNode());
						build_queue.push_back(std::pair(left_children, current_node.second->GetChild(accel::Child::Left)));
					}
					else
					{
						nodes[current_node.first].first = (unsigned int)triangles.size();
						triangles.insert(triangles.end(), current_node.second->GetElements().begin(), current_node.second->GetElements().end());
						nodes[current_node.first].last = (unsigned int)triangles.size();
					}
				}
			}
		}

		details_->scene_.h_scenetree.nodes_num_ = (unsigned int)nodes.size();
		details_->scene_.h_scenetree.nodes_ = new CustomNode<CUDATree, Triangle>[nodes.size()];
		memcpy(details_->scene_.h_scenetree.nodes_, &nodes[0], nodes.size() * sizeof(CustomNode<CUDATree, Triangle>));

		details_->scene_.h_scenetree.triangles_num_ = (unsigned int)triangles.size();
		details_->scene_.h_scenetree.triangles_ = new Triangle[triangles.size()];
		memcpy(details_->scene_.h_scenetree.triangles_, &triangles[0], triangles.size() * sizeof(Triangle));
    }

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
