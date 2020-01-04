/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "common.h"
#include "log.h"
#include "timer.h"
#include "ray.h"
#include "camera.h"
#include "mesh.h"
#include "scene.h"

#include "input.h"
Input g_input;


#if defined(CPU_KERNEL)
 #include "kernels/raytracing/software/cpu_trace.h"
 CpuTrace& g_kernel = CpuTrace::GetInstance();
#elif defined(CUDA_KERNEL)
 #include "kernels/raytracing/cuda/cuda_trace.h"
 CUDATrace& g_kernel = CUDATrace::GetInstance();
#elif defined(OPENGL_KERNEL)
 #include "kernels/rasterization/opengl/opengl_render.h"
 OpenGLRender& g_kernel = OpenGLRender::GetInstance();
#else
 #error "at least one module should be enabled!"
#endif


#if defined(WIN32)
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	case WM_KEYDOWN:
		g_input.keystatus[wParam] = true;
		g_input.pending = true;
		break;
	
	case WM_MOUSEMOVE:
		g_input.mouse.pos.x = LOWORD(lParam);
		g_input.mouse.pos.y = HIWORD(lParam);
		g_input.mouse.buttonstatus[Input::MouseButton::Left] = wParam & MK_LBUTTON;
		g_input.mouse.buttonstatus[Input::MouseButton::Middle] = wParam & MK_MBUTTON;
		g_input.mouse.buttonstatus[Input::MouseButton::Right] = wParam & MK_RBUTTON;
		g_input.pending = true;
		break;

	case WM_PAINT:
		g_kernel.OnPaint();
		break;

	default:
		return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}

	return 0;
}
#endif

void UpdateWindowText(Handle window, const char* text)
{
#if defined(WIN32)
	SetWindowTextA(window, text);
#else
	XStoreName(window->dpy, window->win, text);
#endif
}

bool IsValidHandle(Handle window)
{
#if defined(WIN32)
	return window != nullptr;
#else
	return window && window->win;
#endif
}

Handle TracyCreateWindow(int width, int height)
{
#if defined(WIN32)

	WNDCLASSEXA win_class = {};
	win_class.cbSize = sizeof(WNDCLASSEX);
	win_class.style = CS_OWNDC | CS_VREDRAW | CS_HREDRAW;
	win_class.lpfnWndProc = WindowProc;
	win_class.hInstance = (HINSTANCE)GetModuleHandle(nullptr);
	win_class.hCursor = LoadCursor(nullptr, IDC_ARROW);
	win_class.hIcon = LoadIcon(nullptr, IDI_WINLOGO);
	win_class.lpszClassName = "TracyWindowClass";
	RegisterClassExA(&win_class);

	// no resize / no maximize / no minimize button
	DWORD win_style = (WS_OVERLAPPEDWINDOW ^ (WS_SIZEBOX | WS_MAXIMIZEBOX | WS_MINIMIZEBOX)) | WS_VISIBLE;

	RECT win_rect = { 0, 0, width, height };
	AdjustWindowRectEx(&win_rect, win_style, false, WS_EX_APPWINDOW);

	HWND win_handle = CreateWindowEx(WS_EX_APPWINDOW,
	                                 "TracyWindowClass",
	                                 ".:: Tracy 2.0 ::. (collecting data...)",
	                                 win_style,
	                                 CW_USEDEFAULT,
	                                 CW_USEDEFAULT,
	                                 win_rect.right - win_rect.left,
	                                 win_rect.bottom - win_rect.top,
	                                 nullptr,
	                                 nullptr,
	                                 (HINSTANCE)GetModuleHandle(nullptr),
	                                 nullptr);

#else

	Display* dpy = XOpenDisplay(nullptr);
	
	int ds = DefaultScreen(dpy);
    Window win = XCreateSimpleWindow(dpy, RootWindow(dpy, ds), 0, 0, width, height, 1, BlackPixel(dpy, ds), WhitePixel(dpy, ds));
    XSelectInput(dpy, win, KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | ButtonMotionMask | StructureNotifyMask);
    
    XInternAtom(dpy, "WM_PROTOCOLS", false);
    Atom close_win_msg = XInternAtom(dpy, "WM_DELETE_WINDOW", false);
    XSetWMProtocols(dpy, win, &close_win_msg, 1);

	XSizeHints size_hint;
	size_hint.flags = PMinSize | PMaxSize;
    size_hint.max_width = 0;
    size_hint.min_width = width;
    size_hint.max_height = 0;
    size_hint.min_height = height;
    XSetWMNormalHints(dpy, win, &size_hint);

    XMapWindow(dpy, win);
	XStoreName(dpy, win, ".:: Tracy 2.0 ::. (collecting data...)");

    handle_t* win_handle = new handle_t;
    win_handle->ds = ds;
    win_handle->dpy = dpy;
    win_handle->win = win;

#endif

    return win_handle;
}

void TracyDestroyWindow(Handle window_handle)
{
#if defined(WIN32)
	DestroyWindow(window_handle);
#else
	XDestroyWindow(window_handle->dpy, window_handle->win);
	XCloseDisplay(window_handle->dpy);
#endif
}

void TracyDisplayWindow(Handle window_handle)
{
#if defined(WIN32)
	ShowWindow(window_handle, SW_SHOW);
	SetForegroundWindow(window_handle);
	UpdateWindow(window_handle);
	SetFocus(window_handle);
#endif
}

bool TracyProcessMessages(Handle window_handle)
{
#if defined(WIN32)
	MSG msg;
	if (PeekMessage(&msg, NULL, NULL, NULL, PM_REMOVE | PM_QS_SENDMESSAGE | PM_QS_INPUT | PM_QS_POSTMESSAGE))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);

		return true;
	}

#else

	if (XEventsQueued(window_handle->dpy, QueuedAlready) > 0)
	{
		XEvent e;
		XNextEvent(window_handle->dpy, &e);
		switch (e.type)
		{
		case Expose:
			TracyLog("[X11] Expose Event\n");
			g_kernel.OnPaint();
			break;
		default:
			TracyLog("[X11] %d Event\n", e.type);
			break;
		}
	}

#endif

	return false;
}

void TracyProcessInputs(Scene& scene, Input& input, Handle window_handle, double dt)
{
	if (input.pending)
	{
		if (input.GetKeyStatus(VK_ESCAPE))
		{
			TracyDestroyWindow(window_handle);
			input.ResetKeyStatus(VK_ESCAPE);
		}

		if (input.GetKeyStatus(Input::KeyGroup::Movement))
		{
			float cam_speed = static_cast<float>(dt);

			Camera& camera = scene.GetCamera();
			vec3 new_cam_pos = camera.GetPosition();
			vec3 cam_up = camera.GetUpVector();
			vec3 cam_forward = camera.GetTarget() - camera.GetPosition();
			vec3 cam_right = normalize(cross(cam_forward, cam_up));

			if (input.keystatus['W']) { new_cam_pos += cam_speed * cam_forward; }

			if (input.keystatus['S']) { new_cam_pos -= cam_speed * cam_forward; }

			if (input.keystatus['A']) { new_cam_pos -= cam_speed * cam_right; }

			if (input.keystatus['D']) { new_cam_pos += cam_speed * cam_right; }

			if (input.keystatus['Q']) { new_cam_pos -= cam_speed * cam_up; }

			if (input.keystatus['E']) { new_cam_pos += cam_speed * cam_up; }

			input.ResetKeyStatus(Input::KeyGroup::Movement);

			camera.UpdateView(new_cam_pos, camera.GetTarget(), cam_up);
			camera.SetDirty(true);
		}

		static bool mousemoving = false;
		if (input.mouse.buttonstatus[Input::MouseButton::Left])
		{
			float cam_speed = static_cast<float>(dt);

			Camera& camera = scene.GetCamera();
			vec3 cam_pos = camera.GetPosition();
			vec3 cam_up = camera.GetUpVector();
			vec3 cam_forward = camera.GetTarget() - camera.GetPosition();
			vec3 cam_right = normalize(cross(cam_forward, cam_up));

			static vec2 oldpos;
			if (!mousemoving)
			{
				oldpos = input.mouse.pos;
				mousemoving = true;
			}

			vec2 delta = cam_speed * (input.mouse.pos - oldpos);

			mat4 rotation(1.f);
			rotation = rotate(rotation, radians(delta.x), cam_up);
			rotation = rotate(rotation, radians(delta.y), cam_right);

			camera.UpdateView((vec4(cam_pos, 1.f) * rotation).xyz, camera.GetTarget(), cam_up);
			camera.SetDirty(true);
		}
		else
		{
			mousemoving = false;
		}

		input.pending = false;
	}
}

bool ShouldQuit(Handle window_handle)
{
#if defined(WIN32)
	MSG msg;
	return (PeekMessage(&msg, NULL, NULL, NULL, PM_NOREMOVE) && msg.message == WM_QUIT);
#else
	static const Atom WM_PROTOCOL = XInternAtom(window_handle->dpy, "WM_PROTOCOLS", false);
    static const Atom close_win_msg = XInternAtom(window_handle->dpy, "WM_DELETE_WINDOW", false);

	if (XEventsQueued(window_handle->dpy, QueuedAlready) > 0)
	{
		XEvent e;
		XPeekEvent(window_handle->dpy, &e);
		return (e.type == KeyPress && (XLookupKeysym(&e.xkey, 0) == XK_Escape)) ||
		       (e.type == ClientMessage && ((Atom)e.xclient.message_type == WM_PROTOCOL && (Atom)e.xclient.data.l[0] == close_win_msg));
	}
	return false;
#endif
}

#if defined(WIN32)
int WINAPI WinMain(HINSTANCE /* hInstance */, HINSTANCE /* hPrevInstance */, LPSTR /* lpCmdLine */, int /* nCmdShow */)
{
	int argc = __argc;
	char** argv = __argv;
#else
int main(int argc, char** argv)
{
#endif

	int WIDTH = 640;
	int HEIGHT = 480;
	char SCENE_PATH[MAX_PATH] = "data/default.scn";
	if (argc == 2)
	{
		memset(SCENE_PATH, 0, MAX_PATH);
		strncpy(SCENE_PATH, argv[1], MAX_PATH);
	}

	Scene world;
	if (world.Init(SCENE_PATH, WIDTH, HEIGHT))
	{
		Handle win_handle = TracyCreateWindow(WIDTH, HEIGHT);
		if (IsValidHandle(win_handle))
		{
			TracyDisplayWindow(win_handle);

			g_kernel.Initialize(win_handle, WIDTH, HEIGHT, world);

			int frame_count = 0;
			Timer trace_timer;
			Timer frame_timer;

			// TODO: threads
			while (!ShouldQuit(win_handle))
			{
				TracyProcessMessages(win_handle);
				
				TracyProcessInputs(world, g_input, win_handle, frame_timer.GetDuration());
				
				frame_timer.Reset();
				frame_timer.Begin();

				trace_timer.Begin();

				g_kernel.UpdateScene();

				g_kernel.RenderScene();

				trace_timer.End();

				++frame_count;

				if (trace_timer.GetDuration() > 1.f)
				{
					int raycount = g_kernel.GetRayCount();
					bool has_ray_count = raycount > 0;

					static char window_title[MAX_PATH];
					snprintf(window_title,
					         MAX_PATH,
					         ".:: Tracy 2.0 (%s) ::. '%s' :: %dx%d@%dspp :: [%d objs] [%d tris] [%.2f %s]",
					         g_kernel.GetName(),
					         world.GetName().c_str(),
					         WIDTH,
					         HEIGHT,
					         g_kernel.GetSamplesPerPixel(),
					         world.GetObjectCount(),
					         world.GetTriCount(),
					         (has_ray_count ? (raycount * 1e-6f) / trace_timer.GetDuration() : frame_count / trace_timer.GetDuration()),
					         (has_ray_count ? "MRays/s" : "fps"));

					UpdateWindowText(win_handle, window_title);

					g_kernel.ResetRayCount();
					trace_timer.Reset();
					frame_count = 0;
				}

				frame_timer.End();
			}
			g_kernel.Shutdown();
			TracyDestroyWindow(win_handle);
		}
		else
		{
			TracyLog("Unable to create window\n");
		}
	}
	else
	{
		TracyLog("Unable to load scene '%s'\n", SCENE_PATH);
	}

	return 0;
}
