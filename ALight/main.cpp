
#include <iostream>
#include "Window.h"
#include "Engine.h"
int main(int argc, char* argv[])
{
	
	Setting::argc = argc;
	Setting::argv = *argv;

	Window::Init(Setting::width, Setting::height);
	Engine::Instance()->RayTracer = new RayTracer(true);
	
	Engine::Instance()->camera = new Camera(make_float3(0, 2, 5), make_float3(0, 1, 0), make_float3(0, 1, 0), 60,
	                                        float(Setting::width) / float(Setting::height));

	auto a = make_float3(1, 1, 1);
	a *= make_float3(1, 0, 0);
	printf("%f,%f,%f\n", a.x, a.y, a.z);

	Engine::Instance()->RayTracer->Init(Window::Data, Setting::width, Setting::height);
	Window::Show("Hello");

	std::cout << "Hello ALight";
}
