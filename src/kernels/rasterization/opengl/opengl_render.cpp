/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "opengl_render.h"
#include "GL/glew.h"

#include "log.h"
#include "scene.h"
#include "gl_mesh.h"

#if defined(_MSC_VER)
 #define DLLEXPORT extern "C" __declspec(dllexport)
#else
 #define DLLEXPORT extern "C" __attribute__((dllexport))
#endif
DLLEXPORT uint32_t NvOptimusEnablement = 0x00000001;

namespace
{
struct OpenGLDetails
{
	static const char* fullscreen_vs;
	static const char* fullscreen_fs;

	static const char* object_vs;
	static const char* object_fs;

	const char* fullscreen_texture_name = "fsTex";
	GLint fullscreen_texture{};
	GLuint fullscreen_shader{};
	GLuint object_shader{};

	GLuint fb{};
	GLuint fb_texture{};

	vector<GLMesh> meshes{};
	vector<GLuint> textures{};

	mat4 view{ 1.f };
	mat4 projection{ 1.f };

#if defined(_WIN32)
	HDC hDC;
	HGLRC hRC;
#else
	Display* dpy;
	GLXContext glCtx;
#endif
} render_data_;
}

#define TO_STRING_HELPER(x) #x
#define TO_STRING(x) TO_STRING_HELPER(x)

const char* OpenGLDetails::object_vs = R"vs(
#version 330
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;

uniform struct
{
	mat4 view;
	mat4 projection;
} matrix;

out struct
{
	vec2 texcoords;
} vs;

void main()
{
	vs.texcoords = uv;
	gl_Position = matrix.projection * matrix.view * vec4(position, 1);
}
)vs";

const char* OpenGLDetails::object_fs = R"fs(
#version 330
out vec4 outColor;

in struct
{
	vec2 texcoords;
} vs;

void main()
{
	outColor = vec4(1,1,1,1);
}
)fs";

const char* OpenGLDetails::fullscreen_vs = R"vs(
#version 330
out vec2 texCoords;
void main()
{
    const vec3 vertices[3] = vec3[3](vec3(-1,-1,0), vec3(3,-1,0), vec3(-1,3,0));

	texCoords = 0.5 * vertices[gl_VertexID].xy + vec2(0.5);
    gl_Position = vec4(vertices[gl_VertexID].xy, 0, 1);
}
)vs";


const char* OpenGLDetails::fullscreen_fs =
"#version 330\n"

// exposure
"vec3 exposure(vec3 x) { return x * " TO_STRING(TRACY_EXPOSURE) "; }\n"

// tonemapping function
#if USE_TONEMAP_REINHARD
"vec3 tonemap(vec3 x) { return clamp(x / (1 + x), 0, 1); }\n"
#elif USE_TONEMAP_ACES
"vec3 tonemap(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1); }\n"
#else
"vec3 tonemap(vec3 x) { return x; }\n"
#endif

// main shader body
R"fs(
out vec4 outColor;
in vec2 texCoords;
uniform sampler2D fsTex;

void main()
{
    vec3 color = texture(fsTex, vec2(texCoords.x, texCoords.y)).rgb;
    outColor = vec4(tonemap(exposure(color)), 1);
}
)fs";


#define GLAssert(call) call; OGL::ensure(glGetError(), __FILE__, __LINE__)

namespace OGL
{
static inline void ensure(GLenum val, const char* file, int line)
{
	while (val != GL_NO_ERROR)
	{
		TracyLog("[OpenGL Error] at %s:%d code: 0x%x\n", file, line, val);
		DEBUG_BREAK();
	}
}

static inline auto CheckShaderError = [](GLuint shader)
{
	GLint res;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &res);
	if (res != GL_TRUE)
	{
		char buffer[512];
		glGetShaderInfoLog(shader, array_size(buffer), nullptr, buffer);
		TracyLog("%s", buffer);
		DEBUG_BREAK();
	}
	
	return (res == GL_TRUE);
};

static inline auto CheckProgramError = [](GLuint program)
{
	GLint res;
	glGetProgramiv(program, GL_LINK_STATUS, &res);
	if (res != GL_TRUE)
	{
		char buffer[512];
		glGetProgramInfoLog(program, array_size(buffer), nullptr, buffer);
		TracyLog("%s", buffer);
		DEBUG_BREAK();
	}

	return (res == GL_TRUE);
};
}

bool OpenGLRender::Startup(const WindowHandle in_Window, const Scene& in_Scene)
{
#if defined(_WIN32)

	PIXELFORMATDESCRIPTOR pfd{};
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 24;
	pfd.cDepthBits = 16;
	pfd.iLayerType = PFD_MAIN_PLANE;

	render_data_.hDC = GetDC(in_Window->win);
	GLuint PixelFormat = ChoosePixelFormat(render_data_.hDC, &pfd);
	SetPixelFormat(render_data_.hDC, PixelFormat, &pfd);
	render_data_.hRC = wglCreateContext(render_data_.hDC);

	wglMakeCurrent(render_data_.hDC, render_data_.hRC);

#else

	render_data_.dpy = in_Window->dpy;

	GLint att[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };
	XVisualInfo* vi = glXChooseVisual(in_Window->dpy, 0, att);
	render_data_.glCtx = glXCreateContext(in_Window->dpy, vi, nullptr, GL_TRUE);
	glXMakeCurrent(in_Window->dpy, in_Window->win, render_data_.glCtx);

#endif

	glewExperimental = GL_TRUE;
	bool res = (glewInit() == GLEW_OK);
	if (res)
	{
		TracyLog("OpenGL initialized: %s (%s)\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

#if USE_TONEMAP_SRGB || USE_TONEMAP_ACES || USE_TONEMAP_REINHARD
		GLAssert(glEnable(GL_FRAMEBUFFER_SRGB));
#endif

		GLAssert(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
		GLAssert(glClearDepth(1.0f));
		GLAssert(glEnable(GL_DEPTH_TEST));
		GLAssert(glDepthFunc(GL_LEQUAL));
		GLAssert(glFrontFace(GL_CCW));
		GLAssert(glEnable(GL_CULL_FACE));

		GLAssert(glViewport(0, 0, in_Scene.GetWidth(), in_Scene.GetHeight()));
		
		GLAssert(glMatrixMode(GL_PROJECTION));
		GLAssert(glLoadIdentity());

		GLAssert(glMatrixMode(GL_MODELVIEW));
		GLAssert(glLoadIdentity());

		GLAssert(glGenTextures(1, &render_data_.fb_texture));
		GLAssert(glBindTexture(GL_TEXTURE_2D, render_data_.fb_texture));
		GLAssert(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, in_Window->width, in_Window->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));
		GLAssert(glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_FALSE));
		GLAssert(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
		GLAssert(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
		GLAssert(glBindTexture(GL_TEXTURE_2D, 0));

		GLuint depth;
		glGenRenderbuffers(1, &depth);
		glBindRenderbuffer(GL_RENDERBUFFER, depth);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, in_Window->width, in_Window->height);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

		GLAssert(glGenFramebuffers(1, &render_data_.fb));
		GLAssert(glBindFramebuffer(GL_FRAMEBUFFER, render_data_.fb));
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, render_data_.fb_texture, 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
		GLAssert(res = res && glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
		GLAssert(glBindFramebuffer(GL_FRAMEBUFFER, 0));

		// object ubershaders
		{
			GLAssert(GLuint VS = glCreateShader(GL_VERTEX_SHADER));
			GLAssert(glShaderSource(VS, 1, &render_data_.object_vs, nullptr));
			GLAssert(glCompileShader(VS));
			res = res && OGL::CheckShaderError(VS);

			GLAssert(GLuint FS = glCreateShader(GL_FRAGMENT_SHADER));
			GLAssert(glShaderSource(FS, 1, &render_data_.object_fs, nullptr));
			GLAssert(glCompileShader(FS));
			res = res && OGL::CheckShaderError(FS);

			GLAssert(render_data_.object_shader = glCreateProgram());
			GLAssert(glAttachShader(render_data_.object_shader, VS));
			GLAssert(glAttachShader(render_data_.object_shader, FS));
			GLAssert(glLinkProgram(render_data_.object_shader));
			res = res && OGL::CheckProgramError(render_data_.object_shader);

			GLAssert(glDetachShader(render_data_.object_shader, VS));
			GLAssert(glDetachShader(render_data_.object_shader, FS));
			GLAssert(glDeleteShader(VS));
			GLAssert(glDeleteShader(FS));
		}

		// Fullscreen triangle shaders
		{
			GLAssert(GLuint VS = glCreateShader(GL_VERTEX_SHADER));
			GLAssert(glShaderSource(VS, 1, &render_data_.fullscreen_vs, nullptr));
			GLAssert(glCompileShader(VS));
			res = res && OGL::CheckShaderError(VS);

			GLAssert(GLuint FS = glCreateShader(GL_FRAGMENT_SHADER));
			GLAssert(glShaderSource(FS, 1, &render_data_.fullscreen_fs, nullptr));
			GLAssert(glCompileShader(FS));
			res = res && OGL::CheckShaderError(FS);

			GLAssert(render_data_.fullscreen_shader = glCreateProgram());
			GLAssert(glAttachShader(render_data_.fullscreen_shader, VS));
			GLAssert(glAttachShader(render_data_.fullscreen_shader, FS));
			GLAssert(glLinkProgram(render_data_.fullscreen_shader));
			res = res && OGL::CheckProgramError(render_data_.fullscreen_shader);

			GLAssert(glDetachShader(render_data_.fullscreen_shader, VS));
			GLAssert(glDetachShader(render_data_.fullscreen_shader, FS));
			GLAssert(glDeleteShader(VS));
			GLAssert(glDeleteShader(FS));

			GLAssert(render_data_.fullscreen_texture = glGetUniformLocation(render_data_.fullscreen_shader, render_data_.fullscreen_texture_name));
		}

		// Upload textures
		for (auto& texture : in_Scene.GetTextures())
		{
			if (texture.IsValid())
			{
				GLuint texture_id;
				GLAssert(glGenTextures(1, &texture_id));
				GLAssert(glBindTexture(GL_TEXTURE_2D, texture_id));
				GLAssert(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, texture.GetWidth(), texture.GetHeight(), 0, GL_RGBA, GL_FLOAT, texture.GetPixels()));
				GLAssert(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
				GLAssert(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
				GLAssert(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
				GLAssert(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
				glGenerateMipmap(GL_TEXTURE_2D);
				GLAssert(glBindTexture(GL_TEXTURE_2D, 0));

				render_data_.textures.emplace_back(texture_id);
			}
		}

		// Upload geometry
		for (auto& mesh : in_Scene.GetObjects())
		{
			render_data_.meshes.emplace_back(mesh, in_Scene.GetMaterial(mesh.GetMaterial()));
		}

		render_data_.view = in_Scene.GetCamera().GetView();
		render_data_.projection = in_Scene.GetCamera().GetProjection();
	}

	return res;
}

void OpenGLRender::Shutdown()
{
	GLAssert(glDeleteFramebuffers(1, &render_data_.fb));

#if defined(_WIN32)

	wglMakeCurrent(NULL, NULL);

	wglDeleteContext(render_data_.hRC);

#else

	glXDestroyContext(render_data_.dpy, render_data_.glCtx);

#endif
}

void OpenGLRender::OnUpdate(const Scene& in_Scene, float in_DeltaTime)
{
}

void OpenGLRender::OnEvent(TracyEvent in_Event, const WindowHandle in_Window, const Scene& in_Scene)
{
	switch (in_Event)
	{
	case TracyEvent::eCameraCut:
		render_data_.view = in_Scene.GetCamera().GetView();
		render_data_.projection = in_Scene.GetCamera().GetProjection();
		break;

	default:
		break;
	}
}

void OpenGLRender::OnRender(const WindowHandle in_Window)
{
	if (LIKELY(IsValidWindowHandle(in_Window)))
	{
		// render to texture
		
		GLAssert(glBindFramebuffer(GL_FRAMEBUFFER, render_data_.fb));
		GLAssert(glEnable(GL_DEPTH_TEST));
		GLAssert(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
		GLAssert(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

		GLAssert(glUseProgram(render_data_.object_shader));
		GLAssert(glUniformMatrix4fv(glGetUniformLocation(render_data_.object_shader, "matrix.view"), 1, GL_FALSE, value_ptr(render_data_.view)));
		GLAssert(glUniformMatrix4fv(glGetUniformLocation(render_data_.object_shader, "matrix.projection"), 1, GL_FALSE, value_ptr(render_data_.projection)));
		
		for (const auto& mesh : render_data_.meshes)
		{
			mesh.Draw(render_data_.object_shader);
		}
		GLAssert(glUseProgram(0));
		GLAssert(glBindFramebuffer(GL_FRAMEBUFFER, 0));

		// present full screen triangle + post process
		GLAssert(glDisable(GL_DEPTH_TEST));
		GLAssert(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
		GLAssert(glClear(GL_COLOR_BUFFER_BIT));

		GLAssert(glEnable(GL_TEXTURE_2D));
		GLAssert(glActiveTexture(GL_TEXTURE0));
		GLAssert(glBindTexture(GL_TEXTURE_2D, render_data_.fb_texture));

		GLAssert(glUseProgram(render_data_.fullscreen_shader));

		GLAssert(glUniform1i(render_data_.fullscreen_texture, 0));

		GLAssert(glDrawArrays(GL_TRIANGLES, 0, 3));

		GLAssert(glUseProgram(0));

		GLAssert(glBindTexture(GL_TEXTURE_2D, 0));
		GLAssert(glDisable(GL_TEXTURE_2D));

#if defined(_WIN32)
		SwapBuffers(render_data_.hDC);
#else
		glXSwapBuffers(in_Window->dpy, in_Window->win);
#endif
	}
}
