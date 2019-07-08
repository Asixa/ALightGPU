
#include <iostream>
#include "Window.h"
#include "Engine.h"
#include "Setting.h"
#include "Scene.h"
#include "DeviceManager.h"

int main(int argc, char* argv[])
{
	Setting::argc = argc;
	Setting::argv = *argv;

	PrintDeviceInfo();

	Window::Init(Setting::width, Setting::height);

	HostScene::Instance() ->Load("");
	HostScene::Instance()->Build();

	Engine::Instance()->RayTracer = new RayTracer(true);
	Engine::Instance()->camera = new Camera(make_float3(0, 2, 20), make_float3(0, 1, 0), make_float3(0, 1, 0), 60,
	                                        float(Setting::width) / float(Setting::height));
	Engine::Instance()->RayTracer->Init(Window::Data, Setting::width, Setting::height);
	Window::Show("ALight");

	std::cout << "Hello ALight";
}
