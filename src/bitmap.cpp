/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#include "bitmap.h"

bool Bitmap::Create(WindowHandle ctx, uint32_t w, uint32_t h)
{
	width = w;
	height = h;

#if defined(_WIN32)

	BITMAPINFO bmi;
	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth = width;
	bmi.bmiHeader.biHeight = height;
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 32;
	bmi.bmiHeader.biCompression = BI_RGB;
	bmi.bmiHeader.biSizeImage = width * height * bmi.bmiHeader.biBitCount / 8;
	HDC hdc = CreateCompatibleDC(GetDC(ctx->win));
	bitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&bitmap_bytes, nullptr, 0);

#else

	bitmap_bytes = new uint32_t[w * h];
	bitmap = XCreateImage(ctx->dpy,
		                  DefaultVisual(ctx->dpy, ctx->ds),
		                  DefaultDepth(ctx->dpy, ctx->ds),
		                  ZPixmap,
		                  0,
		                  reinterpret_cast<char*>(bitmap_bytes),
	                      width,
		                  height,
		                  32,
		                  0);

#endif

	return bitmap_bytes != nullptr;
}

void Bitmap::Destroy()
{
#if defined(_WIN32)
	DeleteObject(bitmap);
#else
	XDestroyImage(bitmap);
#endif
}

void Bitmap::SetPixel(uint32_t x, uint32_t y, const vec3& pixel)
{
	uint32_t encoded = (uint8_t)pixel.b       |
	                  ((uint8_t)pixel.g << 8) |
	                  ((uint8_t)pixel.r << 16);

#if defined(_WIN32)
	bitmap_bytes[y * width + x] = encoded;
#else
	XPutPixel(bitmap, x, height - y, encoded);
#endif
}

void Bitmap::Paint(WindowHandle ctx)
{
#if defined(_WIN32)
	PAINTSTRUCT ps;
	RECT rect;
	HDC hdc = BeginPaint(ctx->win, &ps);
	GetClientRect(ctx->win, &rect);

	HDC srcDC = CreateCompatibleDC(hdc);
	SetStretchBltMode(hdc, COLORONCOLOR);
	SelectObject(srcDC, bitmap);
	StretchBlt(hdc, 0, 0, rect.right, rect.bottom, srcDC, 0, 0, width, height, SRCCOPY);
	DeleteObject(srcDC);

	EndPaint(ctx->win, &ps);
#else	
	XPutImage(ctx->dpy, ctx->win, DefaultGC(ctx->dpy, ctx->ds), bitmap, 0, 0, 0, 0, width, height);
#endif
}
