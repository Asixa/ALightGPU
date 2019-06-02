#include "RayTracer.h"
#include "DeviceManager.h"
#include "vec3.h"
void RayTracer::SetPixel(const int x, const int y, Vec3* c) const
{
	// const auto i = Width * 4 * y + x * 4;
	// //Changes[width * y + x]++;
	// (&Data)[i] = c->r() * 255;
	// &Data[i + 1] = c->g() * 255;
	// &Data[i + 2] = c->b() * 255;
	// Data[i + 3] = static_cast<GLbyte>(255);
}

RayTracer::RayTracer() = default;

RayTracer::~RayTracer() = default;

RayTracer::RayTracer(const bool GPU):GPU(GPU)
{
	if (GPU)device_manager = new DeviceManager();
}

void RayTracer::Init(GLbyte* data,int w,int h)
{
	Width = w;
	Height = h;
	Data = data;
	if (GPU)
	{
		device_manager->Init(this);
	}
}

void RayTracer::Render() const
{
	if(GPU)
	{
		device_manager->Run();
	}
}

