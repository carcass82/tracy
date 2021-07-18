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

	bool Create(WindowHandle ctx, uint32_t w, uint32_t h);

	void Destroy();
	
	void SetPixel(uint32_t x, uint32_t y, const vec3& pixel);
	
	void Paint(WindowHandle ctx);


private:

	uint32_t width;
	uint32_t height;

#if defined(_WIN32)
	HBITMAP bitmap{};
#else
	XImage* bitmap{};
#endif

	uint32_t* bitmap_bytes{};

};
