#include "Engine.h"
#include "Scene.h"
#include "Window.h"

Engine* Engine::instance = nullptr;


void Engine::Update()
{
	if(RayTracer->thingsChanged)
	{
		RayTracer->IPR_Quick = true;
	}
	else if(RayTracer->IPR_Quick)
	{
		Window::dx = 0, Window::dy = 0;
		RayTracer->IPR_Quick = false;
		RayTracer->IPR_reset_once = true;
		Window::mouse_last_x = Window::mouse_last_y = -1;
	}
	RayTracer->thingsChanged = false;

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
	RayTracer->thingsChanged = true;
	camera_w+= a*0.01;
	camera_y += b * 0.01*camera_r;
	if (camera_y < 0.1)camera_y = 0.1;
	auto x = cos(camera_w) * camera_r;
	auto z = sin(camera_w) * camera_r;
	camera->Update(make_float3(x, camera_y, z), HostScene::Instance()->lookat);
}

void Engine::OnMouseScroll(int a)
{
	camera_r += a*0.5f;
	OnMouseMove(0, 0);
}

