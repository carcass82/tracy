/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#include "common.h"
#include "log.h"
#include "timer.h"
#include "ray.h"
#include "camera.h"
#include "mesh.h"
#include "scene.h"

WindowHandle g_win_handle{};

#include "input.h"
Input g_input;

#if !defined(CPU_KERNEL) && !defined(CUDA_KERNEL) && !defined(OPENGL_KERNEL) && !defined(CPU_RASTER_KERNEL)
 #error "at least one module should be enabled!"
#else
 #if defined(CPU_KERNEL)
  #include "kernels/raytracing/software/cpu_trace.h"
 #endif
 #if defined(CUDA_KERNEL)
  #include "kernels/raytracing/cuda/cuda_trace.h"
 #endif
 #if defined(OPENGL_KERNEL)
  #include "kernels/raster/opengl/opengl_render.h"
 #endif
 #if defined(CPU_RASTER_KERNEL)
  #include "kernels/raster/cpu/cpu_render.h"
 #endif
#endif

TracyModule* g_kernel{};

enum class TracyKernel
{
	eCPURT,
	eCUDART,
	eOpenGL,
	eCPU,
	eInvalid
};

TracyKernel KernelNameToID(const std::string& kernelname)
{
	if (kernelname == "CPURTX") return TracyKernel::eCPURT;
	else if (kernelname == "CUDA") return TracyKernel::eCUDART;
	else if (kernelname == "OpenGL") return TracyKernel::eOpenGL;
	else if (kernelname == "CPU") return TracyKernel::eCPU;

	return TracyKernel::eInvalid;
}

bool InitializeKernel(const std::string& kernelname)
{
	TracyKernel kernel = KernelNameToID(kernelname);

	switch (kernel)
	{
#if defined(CPU_KERNEL)
	case TracyKernel::eCPURT:
		g_kernel = new CpuTrace();
		break;
#endif
#if defined(CUDA_KERNEL)
	case TracyKernel::eCUDART:
		g_kernel = new CUDATrace();
		break;
#endif
#if defined(OPENGL_KERNEL)
	case TracyKernel::eOpenGL:
		g_kernel = new OpenGLRender();
		break;
#endif
#if defined(CPU_RASTER_KERNEL)
	case TracyKernel::eCPU:
		g_kernel = new CPURender();
		break;
#endif
	default:
		TracyLog("Invalid kernel choice: should be one in %s\n", "["
#if defined(CPU_KERNEL)
			"CPURTX, "
#endif
#if defined(CUDA_KERNEL)
			"CUDA, "
#endif
#if defined(OPENGL_KERNEL)
			"OpenGL, "
#endif
#if defined(CPU_RASTER_KERNEL)
			"CPU"
#endif
			"]");
	};

	return kernel != TracyKernel::eInvalid && g_kernel != nullptr;
}

#if defined(_WIN32)
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
		break; // we handle the painting, do nothing here

	default:
		return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}

	return 0;
}
#endif

void UpdateWindowText(WindowHandle window, const char* text)
{
#if defined(_WIN32)
	SetWindowTextA(window->win, text);
#else
	XStoreName(window->dpy, window->win, text);
#endif
}

WindowHandle TracyCreateWindow(i32 width, i32 height)
{
#if defined(_WIN32)

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

	return CreateWindowHandle(win_handle, width, height);

#else

	Display* dpy = XOpenDisplay(nullptr);
	
	int ds = DefaultScreen(dpy);
    Window win = XCreateSimpleWindow(dpy, RootWindow(dpy, ds), 0, 0, width, height, 1, BlackPixel(dpy, ds), WhitePixel(dpy, ds));
    XSelectInput(dpy, win, KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | ButtonMotionMask | StructureNotifyMask | ExposureMask);
    
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

	XStoreName(dpy, win, ".:: Tracy 2.0 ::. (collecting data...)");

    return CreateWindowHandle(width, height, ds, dpy, win);

#endif
}

void TracyDestroyWindow(WindowHandle window_handle)
{
#if defined(_WIN32)
	DestroyWindow(window_handle->win);
#else
	XUnmapWindow(window_handle->dpy, window_handle->win);
	XDestroyWindow(window_handle->dpy, window_handle->win);
	XCloseDisplay(window_handle->dpy);
#endif

	ReleaseWindowHandle(window_handle);
}

void TracyDisplayWindow(WindowHandle window_handle)
{
#if defined(_WIN32)
	ShowWindow(window_handle->win, SW_SHOW);
	SetForegroundWindow(window_handle->win);
	UpdateWindow(window_handle->win);
	SetFocus(window_handle->win);
#else	
    XMapWindow(window_handle->dpy, window_handle->win);
#endif
}

void TracyUpdateWindow(WindowHandle window_handle)
{
#if defined(_WIN32)
	InvalidateRect(window_handle->win, nullptr, FALSE);
	UpdateWindow(window_handle->win);
#else
	XClearArea(window_handle->dpy, window_handle->win, 0, 0, 1, 1, true);
	XFlush(window_handle->dpy);
#endif

	// make sure OnRender is issued from the same thread that created the window
	g_kernel->OnRender(window_handle);
}

void TracyProcessMessages(WindowHandle window_handle)
{
#if defined(_WIN32)

	MSG msg;
	if (PeekMessage(&msg, NULL, NULL, NULL, PM_REMOVE | PM_QS_SENDMESSAGE | PM_QS_INPUT | PM_QS_POSTMESSAGE))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

#else

	while (XPending(window_handle->dpy))
	{
		XEvent e;
		XNextEvent(window_handle->dpy, &e);
		switch (e.type)
		{
		case Expose:
			g_kernel->OnRender(window_handle);
			break;
		case KeyPress:
			g_input.keystatus[XLookupKeysym(&e.xkey, 0)] = true;
			g_input.pending = true;
			break;
		case ButtonPress:
		case ButtonRelease:
			g_input.mouse.buttonstatus[Input::MouseButton::Left] = (e.xbutton.button == Button1);
			g_input.mouse.buttonstatus[Input::MouseButton::Middle] = (e.xbutton.button == Button2);
			g_input.mouse.buttonstatus[Input::MouseButton::Right] = (e.xbutton.button == Button3);
			[[fallthrough]];
		case MotionNotify:
			g_input.mouse.pos.x = e.xbutton.x;
			g_input.mouse.pos.y = e.xbutton.y;
			g_input.pending = true;
			break;
		}
	}

#endif
}

bool TracyProcessInputs(Scene& scene, Input& input, WindowHandle window_handle, float dt)
{
	bool camera_cut{ false };

	if (input.pending)
	{
		if (input.GetKeyStatus(Input::ESC))
		{
			TracyDestroyWindow(window_handle);
			input.ResetKeyStatus(Input::ESC);
		}

		if (input.GetKeyStatus(Input::KeyGroup::Movement))
		{
			Camera& camera = scene.GetCamera();
			vec3 new_cam_pos = camera.GetPosition();
			vec3 cam_up = camera.GetUpVector();
			vec3 cam_forward = camera.GetTarget() - camera.GetPosition();
			vec3 cam_right = normalize(cross(cam_forward, cam_up));

			if (input.keystatus[Input::W]) { new_cam_pos += dt * cam_forward; }

			if (input.keystatus[Input::S]) { new_cam_pos -= dt * cam_forward; }

			if (input.keystatus[Input::A]) { new_cam_pos -= dt * cam_right; }

			if (input.keystatus[Input::D]) { new_cam_pos += dt * cam_right; }

			if (input.keystatus[Input::Q]) { new_cam_pos -= dt * cam_up; }

			if (input.keystatus[Input::E]) { new_cam_pos += dt * cam_up; }

			input.ResetKeyStatus(Input::KeyGroup::Movement);

			camera.UpdateView(new_cam_pos, camera.GetTarget(), cam_up);
			camera_cut = true;
		}

		static bool mousemoving = false;
		if (input.mouse.buttonstatus[Input::MouseButton::Left])
		{
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

			vec2 delta = dt * (input.mouse.pos - oldpos);

			mat4 rotation(1.f);
			rotation = rotate(rotation, radians(delta.x), cam_up);
			rotation = rotate(rotation, radians(delta.y), cam_right);

			camera.UpdateView((vec4(cam_pos, 1.f) * rotation).xyz, camera.GetTarget(), cam_up);
			camera_cut = true;
		}
		else
		{
			mousemoving = false;
		}

		input.pending = false;
	}

	return camera_cut;
}

bool ShouldQuit(WindowHandle window_handle)
{
#if defined(_WIN32)

	MSG msg;
	return (PeekMessage(&msg, NULL, NULL, NULL, PM_NOREMOVE) && msg.message == WM_QUIT);

#else

	static const Atom WM_PROTOCOL = XInternAtom(window_handle->dpy, "WM_PROTOCOLS", false);
    static const Atom close_win_msg = XInternAtom(window_handle->dpy, "WM_DELETE_WINDOW", false);

	XEvent e;
	XPeekEvent(window_handle->dpy, &e);
	return (e.type == KeyPress && (XLookupKeysym(&e.xkey, 0) == XK_Escape)) ||
	       (e.type == ClientMessage && ((Atom)e.xclient.message_type == WM_PROTOCOL && (Atom)e.xclient.data.l[0] == close_win_msg));

#endif
}

const char* TracySecondsToString(double in_seconds)
{
	static char timestring[64]{};

	u32 seconds = static_cast<u32>(in_seconds);

	u32 minutes = (seconds / 60) % 60;
	seconds -= minutes * 60;

	u32 hours = (minutes / 60) % 60;
	minutes -= hours * 60;

	snprintf(timestring, 64, "%02d:%02d:%02d", hours, minutes, seconds);

	return timestring;
}

template<typename T>
void TracySizeToHumanReadableString(T count, char* out_string, u32 in_size)
{
	static_assert(std::is_arithmetic_v<T>);

	if (count > T(1'000'000'000))
	{
		snprintf(out_string, in_size, "%.2fG", count / 1'000'000'000.0);
	}
	else if (count > T(1'000'000))
	{
		snprintf(out_string, in_size, "%.2fM", count / 1'000'000.0);
	}
	else if (count > T(1'000))
	{
		snprintf(out_string, in_size, "%.2fK", count / 1'000.0);
	}
	else
	{
		snprintf(out_string, in_size, "%d", count);
	}
}

#if defined(_WIN32) && !defined(FORCE_CONSOLE)
int WINAPI WinMain(_In_ HINSTANCE /* hInstance */, _In_opt_ HINSTANCE /* hPrevInstance */, _In_ LPSTR /* lpCmdLine */, _In_ int /* nShowCmd */)
{
	int argc = __argc;
	char** argv = __argv;
#else
int main(int argc, char** argv)
{
#endif

	// defaults, overridden by scene descriptors
	u32 WIDTH = 640;
	u32 HEIGHT = 480;

	char SCENE_PATH[MAX_PATH] = "data/default.scn";

	std::string kernelname{ "CPURTX" };

	for (i32 i = 1; i < argc; ++i)
	{
		std::string argument(argv[i]);

		if (argument == std::string("-scene"))
		{
			if (i + 1 < argc)
			{
				memset(SCENE_PATH, 0, MAX_PATH);
				strncpy(SCENE_PATH, argv[i++ + 1], MAX_PATH);
			}
		}
		else if (argument == std::string("-kernel"))
		{
			if (i + 1 < argc)
			{
				kernelname = argv[i++ + 1];
			}
		}
	}

	if (InitializeKernel(kernelname))
	{
		Scene world;
		if (world.Init(SCENE_PATH, WIDTH, HEIGHT))
		{
			static char object_count[16]{};
			TracySizeToHumanReadableString(world.GetObjectCount(), object_count, 16);

			static char tri_count[16]{};
			TracySizeToHumanReadableString(world.GetTriCount(), tri_count, 16);

			g_win_handle = TracyCreateWindow(WIDTH, HEIGHT);
			if (IsValidWindowHandle(g_win_handle))
			{
				TracyDisplayWindow(g_win_handle);

				if (g_kernel->Startup(g_win_handle, world))
				{
					u32 frame_count = 0;
					Timer trace_timer;
					Timer frame_timer;
					Timer run_timer;

					float avg_raycount = 0;
					float avg_fps = .0f;
					u32 samples = 0;

					run_timer.Begin();

					// TODO: threads
					while (!ShouldQuit(g_win_handle))
					{
						TracyProcessMessages(g_win_handle);

						float dt{ static_cast<float>(frame_timer.GetDuration()) };

						while (g_input.pending)
						{
							if (TracyProcessInputs(world, g_input, g_win_handle, dt))
							{
								g_kernel->OnEvent(TracyEvent::eCameraCut, g_win_handle, world);
							}
						}

						frame_timer.Reset();
						frame_timer.Begin();

						trace_timer.Begin();

						g_kernel->OnUpdate(world, dt);

						trace_timer.End();

						TracyUpdateWindow(g_win_handle);

						++frame_count;

						double trace_duration = trace_timer.GetDuration();
						if (trace_duration > 1.f || frame_count > 100)
						{
							float raycount = g_kernel->GetRayCount(true) / (float)trace_duration;
							float fps = frame_count / (float)trace_duration;

							run_timer.End();

							static char window_title[MAX_PATH] = {};
							snprintf(window_title,
								MAX_PATH,
								".:: Tracy 2.0 (%s) ::. '%s' (%dx%d) %s [%s objs, %s tris][%.2f MRays/s @ %.2f fps]",
								g_kernel->GetModuleName(),
								world.GetName().c_str(),
								WIDTH,
								HEIGHT,
								TracySecondsToString(run_timer.GetDuration()),
								object_count,
								tri_count,
								raycount * 1e-6f,
								fps);

							UpdateWindowText(g_win_handle, window_title);

							++samples;
							avg_raycount = avg_raycount + (raycount - avg_raycount) / (float)samples;
							avg_fps = avg_fps + (fps - avg_fps) / (float)samples;

							trace_timer.Reset();
							frame_count = 0;

							run_timer.Begin();
						}

						frame_timer.End();
					}
					
					g_kernel->Shutdown();
					delete g_kernel;

					TracyDestroyWindow(g_win_handle);

					if (avg_raycount > 0)
					{
						run_timer.End();
						TracyLog("\n*** Performance: %.2f MRays/s and %.2f fps on average - Run time: %s ***\n\n",
							avg_raycount * 1e-6f,
							avg_fps,
							TracySecondsToString(run_timer.GetDuration()));
					}
				}
				else
				{
					TracyLog("Kernel failed to initialize\n");
				}
			}
			else
			{
				TracyLog("Unable to create window\n");
			}
		}
	}
	else
	{
		TracyLog("Unable to initialize kernel '%s'\n", kernelname.c_str());
	}
	
	return 0;
}
