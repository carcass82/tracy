/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "common.h"

#include "timer.h"
#include "ray.h"
#include "camera.h"
#include "mesh.h"
#include "scene.h"

#if CPU_KERNEL
 #include "kernels/raytracing/software/cpu_trace.h"
 CpuTrace& g_kernel = CpuTrace::GetInstance();
#elif OPENGL_KERNEL
 #include "kernels/rasterization/opengl/opengl_render.h"
 OpenGLRender& g_kernel = OpenGLRender::GetInstance();
#elif CUDA_KERNEL
 #include "kernels/raytracing/cuda/cuda_trace.h"
 CUDATrace& g_kernel = CUDATrace::GetInstance();
#else
 #error "at least one module should be enabled!"
#endif

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

static bool IsValidHandle(Handle window)
{
	return window != nullptr;
}

Handle TracyCreateWindow(int width, int height)
{
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

	return win_handle;
}

void TracyDestroyWindow(Handle window_handle)
{
}

void TracyDisplayWindow(Handle window_handle)
{
	ShowWindow(window_handle, SW_SHOW);
	SetForegroundWindow(window_handle);
	UpdateWindow(window_handle);
	SetFocus(window_handle);
}

bool TracyProcessMessages()
{
	MSG msg;
	if (PeekMessage(&msg, NULL, NULL, NULL, PM_REMOVE | PM_QS_SENDMESSAGE | PM_QS_INPUT | PM_QS_POSTMESSAGE))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);

		return true;
	}

	return false;
}

bool ShouldQuit()
{
	MSG msg;
	return (PeekMessage(&msg, NULL, NULL, NULL, PM_NOREMOVE) && msg.message == WM_QUIT);
}

int WINAPI WinMain(HINSTANCE /* hInstance */, HINSTANCE /* hPrevInstance */, LPSTR /* lpCmdLine */, int /* nCmdShow */)
{
	const int WIDTH = 640;
	const int HEIGHT = 480;
	const char* SCENE_PATH = "data/default.scn";
	
	Handle win_handle = TracyCreateWindow(WIDTH, HEIGHT);
	if (IsValidHandle(win_handle))
	{
		TracyDisplayWindow(win_handle);
		char window_title[MAX_PATH];

		Scene world;
		if (world.Init(SCENE_PATH, WIDTH, HEIGHT))
		{
			g_kernel.Initialize(win_handle, WIDTH, HEIGHT, world);

			int frame_count = 0;
			Timer trace_timer;
			Timer frame_timer;

			// TODO: threads
			while (!ShouldQuit())
			{
				if (TracyProcessMessages())
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

					SetWindowTextA(win_handle, window_title);

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
