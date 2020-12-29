/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "cuda_details.h"
#include "cuda_log.h"
#include "scene.h"


#define TO_STRING_HELPER(x) #x
#define TO_STRING(x) TO_STRING_HELPER(x)


const char* RenderData::vs_shader = R"vs(
#version 330
out vec2 texCoords;
void main()
{
    const vec3 vertices[3] = vec3[3](vec3(-1,-1,0), vec3(3,-1,0), vec3(-1,3,0));

	texCoords = 0.5 * vertices[gl_VertexID].xy + vec2(0.5);
    gl_Position = vec4(vertices[gl_VertexID].xy, 0, 1);
}
)vs";


const char* RenderData::fs_shader = 
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
};
}

bool CUDADetails::Initialize(WindowHandle ctx, uint32_t w, uint32_t h)
{
    render_data_.width = w;
    render_data_.height = h;

    InitGLContext(ctx);

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err == GLEW_OK)
    {
        GLAssert(glDisable(GL_DEPTH_TEST));

#if USE_TONEMAP_SRGB || USE_TONEMAP_ACES || USE_TONEMAP_REINHARD
        GLAssert(glEnable(GL_FRAMEBUFFER_SRGB));
#endif

        GLAssert(glClearColor(1.0f, 0.0f, 0.0f, 0.0f));
        GLAssert(glClear(GL_COLOR_BUFFER_BIT));

        GLAssert(glViewport(0, 0, w, h));

        GLAssert(glMatrixMode(GL_PROJECTION));
        GLAssert(glPushMatrix());
        GLAssert(glLoadIdentity());
        GLAssert(glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0));

        GLAssert(glMatrixMode(GL_MODELVIEW));
        GLAssert(glLoadIdentity());

        GLAssert(glGenTextures(1, &render_data_.output_texture));
        GLAssert(glBindTexture(GL_TEXTURE_2D, render_data_.output_texture));
        GLAssert(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr));
        GLAssert(glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_FALSE));
        GLAssert(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        GLAssert(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        GLAssert(glBindTexture(GL_TEXTURE_2D, 0));

        {
            GLAssert(GLuint VS = glCreateShader(GL_VERTEX_SHADER));
            GLAssert(glShaderSource(VS, 1, &render_data_.vs_shader, nullptr));
            GLAssert(glCompileShader(VS));
            OGL::CheckShaderError(VS);

            GLAssert(GLuint FS = glCreateShader(GL_FRAGMENT_SHADER));
            GLAssert(glShaderSource(FS, 1, &render_data_.fs_shader, nullptr));
            GLAssert(glCompileShader(FS));
            OGL::CheckShaderError(FS);

            GLAssert(render_data_.fullscreen_shader = glCreateProgram());
            GLAssert(glAttachShader(render_data_.fullscreen_shader, VS));
            GLAssert(glAttachShader(render_data_.fullscreen_shader, FS));
            GLAssert(glLinkProgram(render_data_.fullscreen_shader));
            OGL::CheckProgramError(render_data_.fullscreen_shader);

            GLAssert(glDetachShader(render_data_.fullscreen_shader, VS));
            GLAssert(glDetachShader(render_data_.fullscreen_shader, FS));
            GLAssert(glDeleteShader(VS));
            GLAssert(glDeleteShader(FS));

            GLAssert(render_data_.fullscreen_texture = glGetUniformLocation(render_data_.fullscreen_shader, render_data_.fullscreen_texture_name));
        }

        TracyLog("OpenGL initialized: %s (%s)\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

        return kernel_.Setup(&render_data_);
    }

    return false;
}

void CUDADetails::InitGLContext(WindowHandle ctx)
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

    render_data_.hDC = GetDC(ctx->win);
    GLuint PixelFormat = ChoosePixelFormat(render_data_.hDC, &pfd);
    SetPixelFormat(render_data_.hDC, PixelFormat, &pfd);
    render_data_.hRC = wglCreateContext(render_data_.hDC);
    
    wglMakeCurrent(render_data_.hDC, render_data_.hRC);

#else

    GLint att[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };
    XVisualInfo* vi = glXChooseVisual(win_handle_->dpy, 0, att);
    GLXContext glc = glXCreateContext(win_handle_->dpy, vi, nullptr, GL_TRUE);
    glXMakeCurrent(win_handle_->dpy, win_handle_->win, glc);

#endif
}

bool CUDADetails::ProcessScene(const Scene& scene)
{
    return kernel_.SetupScene(scene);
}

void CUDADetails::Update(const Scene& scene)
{    
    if (camera_updated_)
    {
        kernel_.UpdateCamera(scene.GetCamera());
        camera_updated_ = false;
    }

    kernel_.Trace();
}

void CUDADetails::Render(WindowHandle ctx, uint32_t w, uint32_t h)
{
    GLAssert(glClear(GL_COLOR_BUFFER_BIT));

    GLAssert(glEnable(GL_TEXTURE_2D));
    GLAssert(glActiveTexture(GL_TEXTURE0));
    GLAssert(glBindTexture(GL_TEXTURE_2D, render_data_.output_texture));
    
    GLAssert(glUseProgram(render_data_.fullscreen_shader));

    GLAssert(glUniform1i(render_data_.fullscreen_texture, 0));

    GLAssert(glDrawArrays(GL_TRIANGLES, 0, 3));

    GLAssert(glUseProgram(0));

    GLAssert(glBindTexture(GL_TEXTURE_2D, 0));
    GLAssert(glDisable(GL_TEXTURE_2D));

#if defined(_WIN32)
    SwapBuffers(render_data_.hDC);
#else
    glXSwapBuffers(win_handle_->dpy, win_handle_->win);
#endif
}

void CUDADetails::Shutdown()
{
    kernel_.Shutdown();

    GLAssert(glDeleteProgram(render_data_.fullscreen_shader));
    GLAssert(glDeleteTextures(1, &render_data_.output_texture));

    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(render_data_.hRC);
}

void CUDADetails::CameraUpdated()
{
    camera_updated_ = true;
}

uint32_t CUDADetails::GetRayCount()
{
    return kernel_.GetRayCount();
}

void CUDADetails::ResetRayCount()
{
    kernel_.ResetRayCount();
}
