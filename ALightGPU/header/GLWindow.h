#pragma once
#include "root.h"
#include "ray.h"
#include <fstream>

void Render();
bool keyDown[256];
inline void Resize(int width, int height)
{
	glutReshapeWindow(ImageWidth, ImageHeight);
}


void mainMenu(int i)
{
	printf("Command %d", i);
	//key((unsigned char)i, 0, 0);
}
void InitMenus()
{
	glutCreateMenu(mainMenu);
	glutAddMenuEntry("Reset block [1]", '1');
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}


inline void WindowsUpdate()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(ImageWidth, ImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, PixelData);
	glutSwapBuffers();
	glFlush();
}

inline void TimerProc(int id)
{
	glutPostRedisplay();
	glutTimerFunc(1, TimerProc, 1);
}

void mouse(int button, int state, int x, int y)
{
	int mods;

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

	glutPostRedisplay();
}
void motion(int x, int y)
{
	printf("MouseMove %d,%d", x, y);
}

void key(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	}
	keyDown[key] = true;

	glutPostRedisplay();
}

void keyUp(unsigned char key, int /*x*/, int /*y*/)
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
	glClearColor(1.0, 0.0, 1.0, 1.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, 200.0, 0.0, 150.0);
	//InitMenus();
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(key);
	glutKeyboardUpFunc(keyUp);


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

