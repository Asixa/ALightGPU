
#include <iostream>
#include "Window.h"
#include "Engine.h"
int main(int argc, char* argv[])
{
	
	Setting::argc = argc;
	Setting::argv = *argv;

	Window::Init(512, 512);
	Engine::Instance()->RayTracer = new RayTracer(true);
	
	Engine::Instance()->camera = new Camera(make_float3(0, 2, 5), make_float3(0, 1, 0), make_float3(0, 1, 0), 60,
	                                        float(512) / float(512));
	Engine::Instance()->RayTracer->Init(Window::Data, 512, 512);
	Window::Show("Hello");

	std::cout << "Hello ALight";
}
