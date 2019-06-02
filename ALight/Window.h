#pragma once
#include <GL\glew.h>
#include <GL\freeglut.h>
#include "Setting.h"
#include "RayTracer.h"

class RayTracer;
class Window
{
	static float last_time;
	static void Resize(int width, int height);
	static void WindowsUpdate();
	
public:
	static int Height, Width;
	static GLbyte* Data;
	static float FPS;
	//static RayTracer* RayTracer;
	static void Init(int init_wdith, int init_height);
	static void Show(const char* title);
	static void CaculateFPS();
};



