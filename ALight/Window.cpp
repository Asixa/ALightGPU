#include "Window.h"
#include "Defines.h"
#include <cstdio>
#include "Engine.h"
#include <ostream>
#include <iostream>
#include "Setting.h"


int Window::Height = 0;
int Window::Width = 0;
int Window::dx = 0;
int Window::dy = 0;
int Window::mouse_last_x = -1;
int Window::mouse_last_y = -1;
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
	Engine::Instance()->Update();
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

inline void GlMouseEvent(int button, int state, int x, int y)
{
	int mods;
	if (state == GLUT_DOWN)
	{
		Engine::Instance()->RayTracer->IPR_Quick = true;
	}
	else if (state == GLUT_UP)
	{
		Window::dx = 0, Window::dy = 0;
		Engine::Instance()->RayTracer->IPR_Quick = false;
		Engine::Instance()->RayTracer->IPR_reset_once = true;
		Window::mouse_last_x = Window::mouse_last_y = -1;
	}
	if (button == 4)Engine::Instance()->OnMouseScroll(1);
	else if(button == 3)Engine::Instance()->OnMouseScroll(-1);
}
inline void GlMouseMotion(int x, int y)
{
	if (Window::mouse_last_x == -1)Window::mouse_last_x = x;
	else { Window::dx = x - Window::mouse_last_x;
		Window::mouse_last_x = x; }
	if (Window::mouse_last_y == -1)Window::mouse_last_y = y;
	else { Window::dy = y - Window::mouse_last_y;
		Window::mouse_last_y = y; }
	Engine::Instance()->OnMouseMove(Window::dx, Window::dy);
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

	glutMouseFunc(GlMouseEvent);
	glutMotionFunc(GlMouseMotion);
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

void Window::Savepic()
{
		// std::ofstream outf;
		// outf.open("/Output/abc.ppm");
		// outf << "P3\n" << Setting::width << " " << Setting::height << "\n255\n";
		// for (auto h = ImageHeight - 1; h >= 0; h--)
		// {
		// 	for (int i = 0; i < 3; i += 3)
		// 	{
		// 		outf << PixelData[h * (rgbwidth)+(i + 0)] << " " <<
		// 			PixelData[h * (rgbwidth)+(i + 1)] << " " <<
		// 			PixelData[h * (rgbwidth)+(i + 2)] << " \n";
		// 	}
		// }
		// outf.close();
		// std::cout << "finished" << std::endl;
}
