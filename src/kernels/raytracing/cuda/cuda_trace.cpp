/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "cuda_trace.h"
#include "cuda_details.h"
#include "log.h"

namespace
{
    CUDADetails Details;
}

int CUDATrace::GetRayCount() const
{
    return Details.GetRayCount();
}

void CUDATrace::ResetRayCount()
{
    Details.ResetRayCount();
}

bool CUDATrace::Startup(const WindowHandle in_Window, const Scene& in_Scene)
{
    return Details.Initialize(in_Window, in_Window->width, in_Window->height) &&
           Details.ProcessScene(in_Scene);
}

void CUDATrace::Shutdown()
{
    Details.Shutdown();
}

void CUDATrace::OnUpdate(const Scene& in_Scene, float in_DeltaTime)
{
    Details.Update(in_Scene);
}

void CUDATrace::OnEvent(TracyEvent in_Event, const WindowHandle in_Window, const Scene& in_Scene)
{
    switch (in_Event)
    {
    case TracyEvent::eCameraCut:
        Details.CameraUpdated();
        break;
    default:
        break;
    }
}

void CUDATrace::OnRender(const WindowHandle in_Window)
{
    if (LIKELY(IsValidWindowHandle(in_Window)))
    {
        Details.Render(in_Window, in_Window->width, in_Window->height);
    }
}
