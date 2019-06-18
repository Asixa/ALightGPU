#pragma once
#include <GL\glew.h>
#include <GL\freeglut.h>

class RayTracer;
class Window
{
	static float last_time;
	static void Resize(int width, int height);
	static void WindowsUpdate();
	
public:
	static int Height, Width;
	static int  dx, dy, mouse_last_x, mouse_last_y;
	static GLbyte* Data;
	static float FPS;
	//static RayTracer* RayTracer;
	static void Init(int init_wdith, int init_height);
	static void Show(const char* title);
	static void CaculateFPS();
	void Savepic();
};



