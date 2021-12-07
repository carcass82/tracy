/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#include "bitmap.h"

bool Bitmap::Create(WindowHandle ctx, u32 w, u32 h)
{
	width_ = w;
	height_ = h;

#if defined(_WIN32)

	BITMAPINFO bmi;
	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth = width_;
	bmi.bmiHeader.biHeight = height_;
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 32;
	bmi.bmiHeader.biCompression = BI_RGB;
	bmi.bmiHeader.biSizeImage = width_ * height_ * bmi.bmiHeader.biBitCount / 8;
	HDC hdc = CreateCompatibleDC(GetDC(ctx->win));
	bitmap_ = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&bitmap_bytes_, nullptr, 0);

#else

	bitmap_bytes_ = new u32[w * h];
	bitmap_ = XCreateImage(ctx->dpy,
		                   DefaultVisual(ctx->dpy, ctx->ds),
		                   DefaultDepth(ctx->dpy, ctx->ds),
		                   ZPixmap,
		                   0,
		                   reinterpret_cast<char*>(bitmap_bytes_),
	                       width_,
		                   height_,
		                   32,
		                   0);

#endif

	return bitmap_bytes_ != nullptr;
}

void Bitmap::Destroy()
{
#if defined(_WIN32)
	DeleteObject(bitmap_);
#else
	XDestroyImage(bitmap_);
#endif
}

void Bitmap::SetPixel(u32 x, u32 y, const vec3& pixel)
{
	u32 encoded =  (u8)pixel.b       |
	              ((u8)pixel.g << 8) |
	              ((u8)pixel.r << 16);

#if defined(_WIN32)
	bitmap_bytes_[y * width_ + x] = encoded;
#else
	XPutPixel(bitmap_, x, height_ - y, encoded);
#endif
}

void Bitmap::Clear(const vec3& color)
{
	u32 encoded = (u8)color.b       |
	             ((u8)color.g << 8) |
	             ((u8)color.r << 16);

	for (u32 i = 0; i < width_ * height_; ++i)
	{
		bitmap_bytes_[i] = encoded;
	}
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
	SelectObject(srcDC, bitmap_);
	StretchBlt(hdc, 0, 0, rect.right, rect.bottom, srcDC, 0, 0, width_, height_, SRCCOPY);
	DeleteObject(srcDC);

	EndPaint(ctx->win, &ps);
#else	
	XPutImage(ctx->dpy, ctx->win, DefaultGC(ctx->dpy, ctx->ds), bitmap_, 0, 0, 0, 0, width_, height_);
#endif
}
