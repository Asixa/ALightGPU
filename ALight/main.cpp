
#include <iostream>
#include "Window.h"
#include "Scene.h"
#include "Engine.h"
#include "Setting.h"

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

	Engine::Instance()->RayTracer->Init(Window::Data, Setting::width, Setting::height);
	Window::Show("ALight");

	std::cout << "Hello ALight";
}
