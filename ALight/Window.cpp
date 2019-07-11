#include "Window.h"
#include "Defines.h"
#include <cstdio>
#include "Engine.h"
#include "Setting.h"
#include "fstream"
#include <iostream>
#include <ctime>
#include <regex>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
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
	// if (state == GLUT_DOWN)Engine::Instance()->RayTracer->IPR_Quick = true;
	// else if (state == GLUT_UP)
	// {
	// 	Window::dx = 0, Window::dy = 0;
	// 	Engine::Instance()->RayTracer->IPR_Quick = false;
	// 	Engine::Instance()->RayTracer->IPR_reset_once = true;
	// 	Window::mouse_last_x = Window::mouse_last_y = -1;
	// }
	if (button == 4)Engine::Instance()->OnMouseScroll(1);
	else if(button == 3)Engine::Instance()->OnMouseScroll(-1);
}
inline void GlMouseMotion(int x, int y)
{
	if (Window::mouse_last_x == -1)Window::mouse_last_x = x;
	else { 
		Window::dx = x - Window::mouse_last_x;
		Window::mouse_last_x = x;
	}
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
		std::string title = "ALightGPU - "+ std::to_string(Engine::Instance()->RayTracer->sampled);
		// char title[35];
		// printf(title, sizeof(title), "ALightGPU");
		glutSetWindowTitle(title.data());
		FPS = 0;
	}
}
int saveScreenshot(const char* filename)
{
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	int x = viewport[0];
	int y = viewport[1];
	int width = viewport[2];
	int height = viewport[3];
	char* data = (char*)malloc((size_t)(width * height * 3)); // 3 components (R, G, B)
	if (!data)return 0;
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);
	int saved = stbi_write_png(filename, width, height, 3, data, 0);
	free(data);
	return saved;
}
void Window::Savepic()
{
	time_t curr_time;
	tm* curr_tm;
	char date_string[100];
	char time_string[100];
	time(&curr_time);
	curr_tm = localtime(&curr_time);
	strftime(date_string, 50, "%B_%d_%Y", curr_tm);
	strftime(time_string, 50, "_%T", curr_tm);
	std::string path = date_string;
	path += std::regex_replace(time_string, std::regex(":"), "_");;
	stbi_flip_vertically_on_write(1);
	//saveScreenshot(("output/" + path + "_screenshot.png").data());
	stbi_write_png(("output/" + path + "_result.png").data(), Width, Height, 4, Data, Width * 4);
	
	
	// std::ofstream outf;
	// outf.open("output/" + path + "pic.ppm");
	// outf << "P3\n" << Setting::width << " " << Setting::height << "\n255\n";
	// for (auto h = Height - 1; h >= 0; h--)
	// {
	// 	for (int i = 0; i < 3; i += 3)
	// 	{
	// 		outf << Data[h * (3) + (i + 0)] << " " <<
	// 			Data[h * (3) + (i + 1)] << " " <<
	// 			Data[h * (3) + (i + 2)] << " \n";
	// 	}
	// }
	// outf.close();

	std::cout << "Finished Save Picture " <<path<< std::endl;
}
