#include "RayTracer.h"
#include "DeviceManager.h"


#include "Setting.h"
#include <direct.h>
#include <string>
#include <algorithm>
#include "Scene.h"
RayTracer::RayTracer() = default;

RayTracer::~RayTracer() = default;

void RayTracer::ReSetIPR()
{
	if (!Setting::IPR)return;
	sampled = 0;
}

RayTracer::RayTracer(const bool GPU):GPU(GPU)
{
	if (GPU)device_manager = new DeviceManager();
}

void RayTracer::Init(GLbyte* d,int w,int h)
{
	width = w;
	height = h;
	data = d;



	if (GPU)
	{
		device_manager->Init(this,*HostScene::Instance());
	}
}

void RayTracer::Render() const
{
	if(GPU)
	{
		device_manager->Run();
	}
}

