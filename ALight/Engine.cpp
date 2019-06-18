#include "Engine.h"

Engine* Engine::instance = nullptr;


void Engine::Update()
{
	if (RayTracer->IPR_Quick)RayTracer->ReSetIPR();
	if (RayTracer->IPR_reset_once)
	{
		RayTracer->ReSetIPR();
		RayTracer->IPR_reset_once = false;
	}
	RayTracer->Render();
}


void Engine::OnMouseMove(int a, int b)
{
	camera_w+= a*0.01;
	camera_y += b * 0.05;
	if (camera_y < 0.1)camera_y = 0.1;
	auto x = cos(camera_w) * camera_r;
	auto z = sin(camera_w) * camera_r;
	camera->Update(make_float3(x, camera_y, z), make_float3(3, 30, 0));
}

void Engine::OnMouseScroll(int a)
{
	camera_r += a*0.1f;
	if(!RayTracer->IPR_Quick)OnMouseMove(0, 0);
}

