#pragma once
#include "root.h"
#include "ray.h"
#include <fstream>
#include <string>

void Render();
void ReSetIPR();
void OnMouseMove(int x, int y);
void OnKeyDown();


namespace GLWindow {
	static float framesPerSecond = 0.0f;       // This will store our fps
	static float lastTime = 0.0f;       // This will hold the time from the last frame
	bool keyDown[256];
	bool IPR_reset,IPR_Reset_once;
	int mouse_last_x = -1, mouse_last_y = -1,dx=0,dy=0;

	inline void Resize(int width, int height)
	{
		glutReshapeWindow(ImageWidth, ImageHeight);
	}

	inline void MainMenu(int i)
	{
		printf("Command %d", i);
		//GLKeyDownEvent((unsigned char)i, 0, 0);
	}

	inline void InitMenus()
	{
		glutCreateMenu(MainMenu);
		glutAddMenuEntry("Reset block [1]", '1');
		glutAttachMenu(GLUT_RIGHT_BUTTON);
	}

	inline void CaculateFPS()
	{
		const auto current_time = GetTickCount() * 0.001f;
		++framesPerSecond;
		if (current_time - lastTime > 1.0f)
		{
			lastTime = current_time;
			char title[35];
			snprintf(title, sizeof(title), "ALightGPU  FPS:%d SPP:%d", int(framesPerSecond), current_spp);
			glutSetWindowTitle(title);
			framesPerSecond = 0;
		}
	}

	inline void WindowsUpdate()
	{
		
		glClear(GL_COLOR_BUFFER_BIT);
		CaculateFPS();
		glDrawPixels(ImageWidth, ImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, PixelData);
		glutSwapBuffers();
		glFlush();
	}

	inline void TimerProc(int id)
	{
		if (IPR_reset)ReSetIPR();
		if(IPR_Reset_once)
		{
			ReSetIPR();
			IPR_Reset_once = false;
		}
		Render();
		glutPostRedisplay();
		glutTimerFunc(1, TimerProc, 1);
	}

	inline void GlMouseEvent(int button, int state, int x, int y)
	{
		int mods;
		if (state == GLUT_DOWN)
		{
			IPR_reset = true;
		}
		else if (state == GLUT_UP)
		{
			dx = 0, dy = 0;
			IPR_reset = false;
			IPR_Reset_once = true;
			mouse_last_x = mouse_last_y = -1;
		}

		// if (state == GLUT_DOWN)
		// {
		// 	buttonState |= 1 << button;
		// }
		// else if (state == GLUT_UP)
		// {
		// 	buttonState = 0;
		// }
		//
		// mods = glutGetModifiers();
		//
		// if (mods & GLUT_ACTIVE_SHIFT)
		// {
		// 	buttonState = 2;
		// }
		// else if (mods & GLUT_ACTIVE_CTRL)
		// {
		// 	buttonState = 3;
		// }
		//
		// ox = x;
		// oy = y;
		//
		// if (displaySliders)
		// {
		// 	if (params->Mouse(x, y, button, state))
		// 	{
		// 		glutPostRedisplay();
		// 		return;
		// 	}
		// }

		//glutPostRedisplay();
	}

	inline void GlMouseMotion(int x, int y)
	{
		
		if (mouse_last_x == -1)mouse_last_x = x;
		else { dx = x - mouse_last_x; mouse_last_x = x; }
		if (mouse_last_y == -1)mouse_last_y = y;
		else { dy = y - mouse_last_y; mouse_last_y = y; }
		OnMouseMove(dx, dy);
		//IPR_reset = true;
	}

	inline void GLKeyDownEvent(unsigned char key, int /*x*/, int /*y*/)
	{
		OnKeyDown();

		switch (key)
		{
		default:;
		}
		keyDown[key] = true;

	}

	inline void GLKeyUpEvent(unsigned char key, int /*x*/, int /*y*/)
	{
		keyDown[key] = false;
	}

	inline void InitWindow(int argc, char** argv, unsigned int mode, int x_position, int y_position, int width, int heigth, const char * title)
	{
		glutInit(&argc, argv);
		glutInitDisplayMode(mode);
		glutInitWindowPosition(x_position, y_position);
		glutInitWindowSize(width, heigth);
		glutCreateWindow(title);
		glClearColor(0, 0.0, 0, 1.0);
		glMatrixMode(GL_PROJECTION);
		gluOrtho2D(0.0, 200.0, 0.0, 150.0);
		//InitMenus();
		glutMouseFunc(GlMouseEvent);
		glutMotionFunc(GlMouseMotion);
		glutKeyboardFunc(GLKeyDownEvent);
		glutKeyboardUpFunc(GLKeyUpEvent);
		


		glutReshapeFunc(Resize);
		glutTimerFunc(1, TimerProc, 1);
		glutDisplayFunc(WindowsUpdate);
		glutMainLoop();
	}


	inline void Savepic()
	{
		std::ofstream outf;
		outf.open("/Output/abc.ppm");
		outf << "P3\n" << ImageWidth << " " << ImageHeight << "\n255\n";
		for (auto h = ImageHeight - 1; h >= 0; h--)
		{
			for (int i = 0; i < rgbwidth; i += 3)
			{
				outf << PixelData[h *(rgbwidth)+(i + 0)] << " " <<
					PixelData[h *(rgbwidth)+(i + 1)] << " " <<
					PixelData[h *(rgbwidth)+(i + 2)] << " \n";
			}
		}
		outf.close();
		std::cout << "finished" << std::endl;
	}
}

