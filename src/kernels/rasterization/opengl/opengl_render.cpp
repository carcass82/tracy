/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "opengl_render.h"
#include "materials.h"
#include "GL/glew.h"

#if !defined(WIN32)
#include <GL/glx.h>
#endif

//
// OpenGL - Details
//

struct OpenGLRender::Details
{
	struct GLMesh
	{
		GLMesh(GLuint in_vao, int in_indexcount, const vec3& in_albedo, float in_metalness, float in_roughness, float in_ior)
			: vao{ in_vao }
			, indexcount{ in_indexcount }
			, albedo{ in_albedo }
			, metalness{ in_metalness }
			, roughness{ in_roughness }
			, ior{ in_ior }
		{}

		GLuint vao;
		int indexcount;
		
		vec3 albedo;
		float metalness;
		float roughness;
		float ior;
	};
	vector<GLMesh> meshes;

	struct Light
	{
		Light(const vec3& in_pos, const vec3& in_color)
			: position{ in_pos }
			, color{ in_color }
			, attenuation{ 1.0f, 0.09f, 0.032f }
		{}

		vec3 position;
		vec3 color;
		vec3 attenuation;
	};
	vector<Light> lights;

	GLuint vs;
	GLuint fs;
	GLuint shader;
};


//
// OpenGL Render
//

OpenGLRender::OpenGLRender()
	: details_(new Details)
{
}

OpenGLRender::~OpenGLRender()
{
	for (const Details::GLMesh& mesh : details_->meshes)
	{
		glDeleteVertexArrays(1, &mesh.vao);
	}
	delete details_;
}

void OpenGLRender::Initialize(Handle in_window, int in_width, int in_height, const Scene& in_scene)
{
	win_handle_ = in_window;
	win_width_ = in_width;
	win_height_ = in_height;

	camera_ = &in_scene.GetCamera();

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

		glEnable(GL_DEPTH_TEST);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClearDepth(1.0f);
		
		glViewport(0, 0, win_width_, win_height_);

		glEnable(GL_CULL_FACE);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Upload geometry
		for (const Mesh& mesh : in_scene.GetObjects())
		{
			if (mesh.GetMaterial()->GetType() != Material::MaterialID::eEMISSIVE)
			{
				GLuint vao;
				glGenVertexArrays(1, &vao);
				glBindVertexArray(vao);

				GLuint vb;
				glGenBuffers(1, &vb);
				glBindBuffer(GL_ARRAY_BUFFER, vb);
				glBufferData(GL_ARRAY_BUFFER, mesh.GetVertexCount() * sizeof(Vertex), &mesh.GetVertices()[0], GL_STATIC_DRAW);

				GLuint ib;
				glGenBuffers(1, &ib);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.GetIndices().size() * sizeof(Index), &mesh.GetIndices()[0], GL_STATIC_DRAW);

				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, pos));
				glEnableVertexAttribArray(0);

				glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, normal));
				glEnableVertexAttribArray(1);

				glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, uv0));
				glEnableVertexAttribArray(2);

				glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, tangent));
				glEnableVertexAttribArray(3);

				glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, bitangent));
				glEnableVertexAttribArray(4);

				glBindVertexArray(0);

				glDeleteBuffers(1, &vb);
				glDeleteBuffers(1, &ib);

				details_->meshes.emplace_back(vao,
					                          mesh.GetIndexCount(),
					                          mesh.GetMaterial()->GetAlbedo(),
					                          mesh.GetMaterial()->GetType() == Material::MaterialID::eMETAL ? 1.f : .0f,
				                              mesh.GetMaterial()->GetRoughness(),
				                              mesh.GetMaterial()->GetIOR());
			}
			else
			{
				// consider emissive objects as lights
				details_->lights.emplace_back(mesh.GetCenter(), mesh.GetMaterial()->GetAlbedo());
			}
		}

		// Create Shaders
		{
			details_->vs = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(details_->vs, 1, &shaders::vs_source, NULL);
			glCompileShader(details_->vs);

			details_->fs = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(details_->fs, 1, &shaders::fs_source, NULL);
			glCompileShader(details_->fs);

#if defined _DEBUG
			static char buffer[512];
			GLint status;

			glGetShaderiv(details_->vs, GL_COMPILE_STATUS, &status);
			if (status != GL_TRUE)
			{
				glGetShaderInfoLog(details_->vs, array_size(buffer), nullptr, buffer);
				DEBUG_BREAK();
			}

			glGetShaderiv(details_->fs, GL_COMPILE_STATUS, &status);
			if (status != GL_TRUE)
			{
				glGetShaderInfoLog(details_->fs, array_size(buffer), nullptr, buffer);
				DEBUG_BREAK();
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
                DEBUG_BREAK();
            }
#endif
		}

		// map UBO
		{
			GLuint ubo_matrices_idx = glGetUniformBlockIndex(details_->shader, "matrices");
			glUniformBlockBinding(details_->shader, ubo_matrices_idx, 0);

			GLuint ubo_matrices;
			glGenBuffers(1, &ubo_matrices);
			glBindBuffer(GL_UNIFORM_BUFFER, ubo_matrices);
			glBufferData(GL_UNIFORM_BUFFER, 2 * sizeof(mat4), NULL, GL_STATIC_DRAW);
			glBufferSubData(GL_UNIFORM_BUFFER, sizeof(mat4) * 0, sizeof(mat4), value_ptr(camera_->GetProjection()));
			glBufferSubData(GL_UNIFORM_BUFFER, sizeof(mat4) * 1, sizeof(mat4), value_ptr(camera_->GetView()));
			glBindBuffer(GL_UNIFORM_BUFFER, 0);
			glBindBufferRange(GL_UNIFORM_BUFFER, 0, ubo_matrices, 0, 2 * sizeof(mat4));
		}

		init_ = true;
	}
}

void OpenGLRender::RenderScene()
{
	if (IsInitialized())
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//
		// debug - show wireframe
		//
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		glUseProgram(details_->shader);

		for (int i = 0; i < static_cast<int>(details_->meshes.size()); ++i)
		{
			glUniform3fv(glGetUniformLocation(details_->shader, "material.albedo"), 1, value_ptr(details_->meshes[i].albedo));
			glUniform1f(glGetUniformLocation(details_->shader, "material.metalness"), details_->meshes[i].metalness);
			glUniform1f(glGetUniformLocation(details_->shader, "material.roughness"), details_->meshes[i].roughness);
			glUniform1f(glGetUniformLocation(details_->shader, "material.ior"), details_->meshes[i].ior);

			// TODO: support multiple lights, avoid setting uniform every time
			glUniform3fv(glGetUniformLocation(details_->shader, "light.position"), 1, value_ptr(details_->lights[0].position));
			glUniform3fv(glGetUniformLocation(details_->shader, "light.color"), 1, value_ptr(details_->lights[0].color));
			glUniform1f(glGetUniformLocation(details_->shader, "light.constant"), details_->lights[0].attenuation.x);
			glUniform1f(glGetUniformLocation(details_->shader, "light.linear"), details_->lights[0].attenuation.y);
			glUniform1f(glGetUniformLocation(details_->shader, "light.quadratic"), details_->lights[0].attenuation.z);

			glBindVertexArray(details_->meshes[i].vao);
			glDrawElements(GL_TRIANGLES, details_->meshes[i].indexcount, GL_UNSIGNED_INT, (GLvoid*)0);
		}
		glBindVertexArray(0);

#if defined(WIN32)
		SwapBuffers(GetDC(win_handle_));
#else
		glXSwapBuffers(win_handle_->dpy, win_handle_->win);
#endif
	}
}
