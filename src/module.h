/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once
#include "common.h"

class Scene;

enum class TracyEvent { eCreate, eResize, eCameraCut, eDestroy };

//
// Base class for Tracy render kernels
//
class TracyModule
{
public:
	
	TracyModule() = default;

	// virtual dtor not used as we're always destroying "T" type objects (but whatever)
	virtual ~TracyModule() {}

	// disable copying
	TracyModule(const TracyModule&) = delete;
	TracyModule& operator=(const TracyModule&) = delete;

	// disable moving (this is not strictly needed but whatever)
	TracyModule(TracyModule&&) = delete;
	TracyModule& operator=(TracyModule&&) = delete;

	// initialize module with provided scene description from main program
	virtual bool Startup(const WindowHandle in_Window, const Scene& in_Scene) = 0;
	
	// free any created resource
	virtual void Shutdown() = 0;
	
	// scene description changed or camera position changed
	virtual void OnUpdate(const Scene& in_Scene, float in_DeltaTime) = 0;
	
	// main program says it's time to render
	virtual void OnRender(const WindowHandle in_Window) = 0;

	// event occurred and module should take appropriate action
	virtual void OnEvent(TracyEvent in_Event, const WindowHandle in_Window, const Scene& in_Scene) = 0;

	// who are you
	virtual const char* GetModuleName() const = 0;

	// get number of rays processed since last reset
	virtual u32 GetRayCount(bool in_ShouldReset = false) { return 0; }
};
