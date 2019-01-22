/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#if defined(_WIN32) && defined(BUILD_GUI)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NOSERVICE
#define NOMCX
#include <Windows.h>

#include <thread>
#include <chrono>
using namespace std::chrono_literals;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

#if USE_GLM
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>
using glm::vec3;
using glm::vec2;
using glm::radians;
using glm::max;
using glm::min;
using glm::clamp;
using glm::lerp;
constexpr float PI = 3.1415926535897932f;
#else
#include "cclib/cclib.h"
using cc::math::vec3;
using cc::math::vec2;
using cc::math::radians;
using cc::util::max;
using cc::util::min;
using cc::util::clamp;
using cc::math::lerp;
using cc::math::PI;
#endif

#include "timer.hpp"
#include "geom.hpp"
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
extern "C" void save_screenshot(int /* w */, int /* h */, vec3* /* pbuffer */);


// CommonData sounds so much better than global variable
struct CommonData
{
    int w;
    int h;
    HBITMAP bitmap;
    vec3* backbuffer;
};
CommonData WinData;


LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_CLOSE:
    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    case WM_KEYDOWN:
        switch (wParam)
        {
        case VK_ESCAPE:
            PostQuitMessage(0);
            break;

        case 'S':
            save_screenshot(WinData.w, WinData.h, WinData.backbuffer);
            break;

        default:
            break;
        }
        break;

    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            RECT rect;
            HDC hdc = BeginPaint(hwnd, &ps);
            GetClientRect(hwnd, &rect);
            
            HDC srcDC = CreateCompatibleDC(hdc);
            SetStretchBltMode(hdc, COLORONCOLOR);
            SelectObject(srcDC, WinData.bitmap);
            StretchBlt(hdc, 0, 0, rect.right, rect.bottom, srcDC, 0, 0, WinData.w, WinData.h, SRCCOPY);
            DeleteObject(srcDC);

            EndPaint(hwnd, &ps);
        }
        break;

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }

    return 0;
}

int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    WNDCLASSEXA windowClass = {};
    windowClass.cbSize = sizeof(WNDCLASSEX);
    windowClass.style = CS_OWNDC | CS_VREDRAW | CS_HREDRAW;
    windowClass.lpfnWndProc = WindowProc;
    windowClass.hInstance = hInstance;
    windowClass.lpszClassName = "TracyWindowClass";
    RegisterClassExA(&windowClass);

    // no resize / no maximize / no minimize button
    DWORD dwStyle = WS_OVERLAPPEDWINDOW ^ (WS_SIZEBOX | WS_MAXIMIZEBOX | WS_MINIMIZEBOX) | WS_VISIBLE;
    const int width = 1024;
    const int height = 768;
    const int samples = 2;

    HWND wHandle = CreateWindowExA(NULL,
                                   "TracyWindowClass",
                                   "Tracy",
                                   dwStyle,
                                   CW_USEDEFAULT,
                                   CW_USEDEFAULT,
                                   width,
                                   height,
                                   NULL,
                                   NULL,
                                   hInstance,
                                   NULL);

    ShowWindow(wHandle, SW_SHOW);
    UpdateWindow(wHandle);
    SetFocus(wHandle);

#if !defined(USE_CUDA)
    Camera cam;
    IShape* world = nullptr;
    setup(cam, float(width) / float(height), &world);
#endif

    vec3* output = new vec3[width * height];
    memset(output, 0, width * height * sizeof(vec3));

    int totrays = 0;
    size_t dummy = 0;

    HBITMAP out_bitmap;
    uint32_t *out_bitmap_bytes;

    BITMAPINFO bmi;
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = height;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    bmi.bmiHeader.biSizeImage = width * height * 4;
    HDC hdc = CreateCompatibleDC(GetDC(0));
    out_bitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&out_bitmap_bytes, NULL, 0x0);

    vec3* output_backbuffer = new vec3[width * height];
    memset(output_backbuffer, 0, width * height * sizeof(vec3));

    WinData.w = width;
    WinData.h = height;
    WinData.bitmap = out_bitmap;
    WinData.backbuffer = output_backbuffer;
    
    double trace_seconds = .0;
    duration<double, std::milli> fps_timer = 0ms;
    constexpr auto ZERO = duration<double, std::milli>(0ms);
    constexpr auto THIRTY_FPS = duration<double, std::milli>(33.3ms);

    bool quit = false;
    MSG msg;
    int frame_count = 0;
    while (!quit)
    {
        if (PeekMessage(&msg, NULL, NULL, NULL, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
            {
                break;
            }

            TranslateMessage(&msg);
            DispatchMessage(&msg);

            continue;
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
            vec3* src = output_backbuffer;
            uint32_t* dst = out_bitmap_bytes;

            for (int j = 0; j < height; ++j)
            {
                for (int i = 0; i < width; ++i)
                {
                    const int index = j * width + i;
                    
                    const vec3 col = output[index] / samples;
                    src[index] = lerp(src[index], col, float(frame_count) / float(frame_count + 1));
                    
                    const vec3 bitmap_col = clamp3(255.99f * sqrtf3(src[index]), .0f, 255.f);

                    *dst++ = (uint8_t)bitmap_col.b | ((uint8_t)bitmap_col.g << 8) | ((uint8_t)bitmap_col.r << 16);
                }
            }
        }

        InvalidateRect(wHandle, NULL, FALSE);
        UpdateWindow(wHandle);

        auto frame_end = high_resolution_clock::now();
        duration<double, std::milli> frame_time = frame_end - frame_start;
        std::this_thread::sleep_for(clamp(THIRTY_FPS - frame_time, ZERO, THIRTY_FPS));

        ++frame_count;
        fps_timer += frame_time;

        // print some random stats every 5 frames
        if (frame_count % 5 == 0)
        {
            static char window_title[MAX_PATH];
            snprintf(window_title,
                MAX_PATH,
                ".:: Tracy (%s) ::. %dx%d@%dspp [%.2f MRays/s - %.2ffps]",
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

            SetWindowTextA(wHandle, window_title);

            totrays = 0;
            trace_seconds = .0;
            fps_timer = 0ms;
        }
    }

    return 0;
}
#endif
