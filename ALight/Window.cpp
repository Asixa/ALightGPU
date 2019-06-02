#include "Window.h"
#include "Defines.h"
#include <cstdio>
#include "Engine.h"
#include <ostream>
#include <iostream>


int Window::Height = 0;
int Window::Width = 0;
float Window::last_time = 0;
float Window::FPS = 0;
GLbyte* Window::Data = new signed char[0];

inline void Window::Resize(int width, int height)
{
	glutReshapeWindow(Width, Height);
}

inline void Window::WindowsUpdate()
{
	//glClear(GL_COLOR_BUFFER_BIT);
	CaculateFPS();
	glDrawPixels(Width, Height, GL_RGBA, GL_UNSIGNED_BYTE, Data);
	glutSwapBuffers();
	glFlush();
}



inline void TimerProc(int id)
{
	// if (IPR_reset)ReSetIPR();
	// if (IPR_Reset_once)
	// {
	// 	ReSetIPR();
	// 	IPR_Reset_once = false;
	// }

	Engine::Instance()->RayTracer->Render();
	glutPostRedisplay();
	glutTimerFunc(1, TimerProc, 1);
}

void Window::Init(int init_wdith, int init_height)
{
	const auto pixel_length = init_wdith * init_height * 4;
	Data = new GLbyte[pixel_length];

	Height = init_height;
	Width = init_wdith;

}

void Window::Show(const char* title)
{

	glutInit(&Setting::argc, &Setting::argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(Width, Height);
	glutCreateWindow(title);
	glClearColor(0, 0.0, 0, 1.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, 200.0, 0.0, 150.0);

	// glutMouseFunc(GlMouseEvent);
	// glutMotionFunc(GlMouseMotion);
	// glutKeyboardFunc(GLKeyDownEvent);
	// glutKeyboardUpFunc(GLKeyUpEvent);

	glutReshapeFunc(Resize);
	glutTimerFunc(1, TimerProc, 1);
	glutDisplayFunc(WindowsUpdate);
	glutMainLoop();
}

void Window::CaculateFPS()
{
	const auto current_time = 0.001f * GetTickCount();
	++FPS;
	if (current_time - last_time > 1.0f)
	{
	
		last_time = current_time;
		char title[35];
		//printf(title, sizeof(title), "ALightGPU");
		glutSetWindowTitle("ALightGPU");
		FPS = 0;
	}
}