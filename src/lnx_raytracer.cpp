/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#if defined(__linux__) && defined(BUILD_GUI)

#include <cstdint>
#include <cstring>
#include <thread>
#include <chrono>
using namespace std::chrono_literals;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include "cclib/cclib.h"
using cc::math::vec3;
using cc::math::vec2;
using cc::math::radians;
using cc::util::clamp;
using cc::util::min;
using cc::util::max;

namespace {
constexpr inline vec3 sqrtf3(const vec3& a) { return vec3{ sqrtf(a.x), sqrtf(a.y), sqrtf(a.z) }; }
constexpr inline vec3 clamp3(const vec3& a, float min, float max) { return vec3{ clamp(a.x, min, max), clamp(a.y, min, max), clamp(a.z, min, max) }; }
}

#include "timer.hpp"

#include "ray.hpp"
#include "materials/material.hpp"
#include "camera.hpp"
#include "shapes/shape.hpp"
#if defined(USE_CUDA)
extern "C" void cuda_trace(int /* w */, int /* h */, int /* ns */, float* /* output */, int& /* totrays */);
#else
extern "C" void setup(Camera& /* cam */, float /* aspect */, IShape** /* world */);
extern "C" void trace(Camera& /* cam */, IShape* /*world*/, int /* w */, int /* h */, int /* ns */, vec3* /* output */, int& /* totrays */, size_t& /* pixel_idx */);
#endif

int main(int argc, char** argv)
{
    Display* dpy = XOpenDisplay(nullptr);
    if (!dpy)
    {
        fputs("cannot connect to x server", stderr);
        return -1;
    }

    constexpr int width = 1024;
    constexpr int height = 768;
    constexpr int samples = 2;

    int ds = DefaultScreen(dpy);
    Window win = XCreateSimpleWindow(dpy, RootWindow(dpy, ds), 0, 0, width, height, 1, BlackPixel(dpy, ds), WhitePixel(dpy, ds));
    XSelectInput(dpy, win, KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | StructureNotifyMask);
    
    const Atom WM_PROTOCOL = XInternAtom(dpy, "WM_PROTOCOLS", false);
    Atom close_win_msg = XInternAtom(dpy, "WM_DELETE_WINDOW", false);
    XSetWMProtocols(dpy, win, &close_win_msg, 1);

    XMapWindow(dpy, win);

#if !defined(USE_CUDA)
    XStoreName(dpy, win, ".:: Tracy (CPU) ::.");
#else
	XStoreName(dpy, win, ".:: Tracy (GPU) ::.");
#endif

#if !defined(USE_CUDA)
    Camera cam;
    IShape* world = nullptr;
    setup(cam, float(width) / float(height), &world);
#endif

    vec3* output = new vec3[width * height];
    memset(output, 0, width * height * sizeof(vec3));

    int totrays = 0;
    size_t dummy = 0;

    vec3* output_backbuffer = new vec3[width * height];
    memset(output_backbuffer, 0, width * height * sizeof(vec3));
    
    uint32_t* data = new uint32_t[width * height];
    XImage* bitmap = XCreateImage(dpy,
                                  DefaultVisual(dpy, ds),
                                  DefaultDepth(dpy, ds),
                                  ZPixmap,
                                  0,
                                  reinterpret_cast<char*>(data),
                                  width,
                                  height,
                                  32,
                                  0);

    double trace_seconds = .0;
    duration<double, std::milli> fps_timer = 0ms;
    constexpr duration<double, std::milli> ZERO(0ms);
    constexpr duration<double, std::milli> THIRTY_FPS(33.3ms);

    bool quit = false;
    XEvent e;
    int frame_count = 0;
    while (!quit)
    {
        if (XPending(dpy))
        {
            XNextEvent(dpy, &e);
            switch (e.type)
            {
            case KeyPress:
                if (XLookupKeysym(&e.xkey, 0) == XK_Escape)
                    quit = true;
                break;

            case KeyRelease:
            case ButtonPress:
            case ButtonRelease:
                break;

            case ClientMessage:
                if ((Atom)e.xclient.message_type == WM_PROTOCOL && (Atom)e.xclient.data.l[0] == close_win_msg)
                    quit = true;
                break;

            case DestroyNotify:
                quit = true;
                break;
                
            case Expose:
                XPutImage(dpy, win, DefaultGC(dpy, ds), bitmap, 0, 0, 0, 0, width, height);
                break;

            default:
                break;
            }
        }

        auto frame_start = high_resolution_clock::now();

        Timer t;
        t.begin();
#if defined(USE_CUDA)
        cuda_trace(width, height, samples, reinterpret_cast<float*>(output), totrays);
#else
        trace(cam, world, width, height, samples, output, totrays, dummy);
#endif
        t.end();
        trace_seconds += t.duration();

        //update bitmap
        {
            vec3* accum = output_backbuffer;

            for (int j = 0; j < height; ++j)
            {
                for (int i = 0; i < width; ++i)
                {
                    const int index = j * width + i;

                    const vec3 col = output[index] / samples;
                    accum[index] = lerp(accum[index], col, float(frame_count) / float(frame_count + 1));

                    const vec3 bitmap_col = clamp3(255.99 * sqrtf3(accum[index]), .0f, 255.f);
                    const uint32_t dst = (uint8_t)bitmap_col.b | ((uint8_t)bitmap_col.g << 8) | ((uint8_t)bitmap_col.r << 16) | (0xff << 24);
                    
                    XPutPixel(bitmap, i, height - j, dst);
                }
            }
        }

        auto frame_end = high_resolution_clock::now();
        duration<double, std::milli> frame_time = frame_end - frame_start;
        std::this_thread::sleep_for(clamp(THIRTY_FPS - frame_time, ZERO, THIRTY_FPS));

        ++frame_count;
        fps_timer += frame_time;

        if (frame_count % 5 == 0)
        {
            static char titlebuf[512];
            snprintf(titlebuf, 512, ".:: Tracy (%s) ::. %dx%d@%dspp [%.2f MRays/s - %.2ffps]",
#if defined(USE_CUDA)
                "GPU",
#else
                "CPU",
#endif
                width,
                height,
                samples,
                (totrays / 1'000'000.0) / trace_seconds,
                5.f / fps_timer.count() * 1000.f);

            XStoreName(dpy, win, titlebuf);

            totrays = 0;
            trace_seconds = .0;
            fps_timer = 0ms;
        }

        XPutImage(dpy, win, DefaultGC(dpy, ds), bitmap, 0, 0, 0, 0, width, height);
        XFlush(dpy);
    }
    XCloseDisplay(dpy);

    return 0;
}
#endif

