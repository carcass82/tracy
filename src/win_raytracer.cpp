/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "common.h"
#if !defined(MAX_PATH)
	#define MAX_PATH 260
#endif

#include "timer.h"
#include "ray.h"
#include "camera.h"
#include "mesh.h"
#include "scene.h"

#if defined(CPU_KERNEL)
 #include "kernels/raytracing/software/cpu_trace.h"
 CpuTrace& g_kernel = CpuTrace::GetInstance();
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
		switch (wParam)
		{
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;

		default:
			break;
		}
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
#if !defined(WIN32)
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

	if (XPending(window_handle->dpy))
	{
		XEvent e;
		XNextEvent(window_handle->dpy, &e);
		switch (e.type)
		{
		case Expose:
			g_kernel.OnPaint();
			break;
		default:
			break;
		}
	}

#endif

	return false;
}

bool ShouldQuit(Handle window_handle)
{
#if defined(WIN32)
	MSG msg;
	return (PeekMessage(&msg, NULL, NULL, NULL, PM_NOREMOVE) && msg.message == WM_QUIT);
#else
	const Atom WM_PROTOCOL = XInternAtom(window_handle->dpy, "WM_PROTOCOLS", false);
    const Atom close_win_msg = XInternAtom(window_handle->dpy, "WM_DELETE_WINDOW", false);

	XEvent e;
	XPeekEvent(window_handle->dpy, &e);
	return (e.type == KeyPress && (XLookupKeysym(&e.xkey, 0) == XK_Escape)) ||
	       (e.type == ClientMessage && ((Atom)e.xclient.message_type == WM_PROTOCOL && (Atom)e.xclient.data.l[0] == close_win_msg));
#endif
}

#if defined(WIN32)
int WINAPI WinMain(HINSTANCE /* hInstance */, HINSTANCE /* hPrevInstance */, LPSTR /* lpCmdLine */, int /* nCmdShow */)
#else
int main(int /* argc */, char** /* argv */)
#endif
{
	const int WIDTH = 640;
	const int HEIGHT = 480;
	const char* SCENE_PATH = "data/default.scn";
	
	Handle win_handle = TracyCreateWindow(WIDTH, HEIGHT);
	if (IsValidHandle(win_handle))
	{
		TracyDisplayWindow(win_handle);
		
		Scene world;
		if (world.Init(SCENE_PATH, WIDTH, HEIGHT))
		{
			g_kernel.Initialize(win_handle, WIDTH, HEIGHT, world);

			int frame_count = 0;
			Timer trace_timer;
			Timer frame_timer;

			// TODO: threads
			while (!ShouldQuit(win_handle))
			{
				if (TracyProcessMessages(win_handle))
				{
					continue;
				}

				frame_timer.Begin();

				g_kernel.UpdateScene();

				g_kernel.RenderScene();
				
				frame_timer.End();

				++frame_count;

				// print some stats every 5 frames
				if (frame_timer.GetDuration() > 1.f)
				{
					bool has_ray_count = g_kernel.GetRayCount() > 0;

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
					         (has_ray_count? (g_kernel.GetRayCount() * 1e-6f) / frame_timer.GetDuration() : frame_count / frame_timer.GetDuration()),
					         (has_ray_count? "MRays/s" : "fps"));

					UpdateWindowText(win_handle, window_title);

					g_kernel.ResetRayCount();
					trace_timer.Reset();
					frame_timer.Reset();
					frame_count = 0;
				}
			}

		}

		g_kernel.Shutdown();

		TracyDestroyWindow(win_handle);
	}

	return 0;
}
