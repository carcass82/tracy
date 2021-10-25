/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once
#include "common.h"

class Bitmap
{
public:

	Bitmap() {}
	Bitmap(WindowHandle ctx, u32 w, u32 h) { Create(ctx, w, h); }
	~Bitmap() { Destroy(); }

	// disable copying
	Bitmap(const Bitmap&) = delete;
	Bitmap& operator=(const Bitmap&) = delete;

	Bitmap(Bitmap&& other) noexcept
		: width{ other.width }, height{ other.height }, bitmap{ std::exchange(other.bitmap, nullptr) }, bitmap_bytes{ std::exchange(other.bitmap_bytes, nullptr) }
	{}

	bool Create(WindowHandle ctx, u32 w, u32 h);

	void Destroy();
	
	void SetPixel(u32 x, u32 y, const vec3& pixel);
	
	void Paint(WindowHandle ctx);

	void Clear(const vec3& color);

private:

	u32 width{};
	u32 height{};

#if defined(_WIN32)
	HBITMAP bitmap{};
#else
	XImage* bitmap{};
#endif

	u32* bitmap_bytes{};

};
