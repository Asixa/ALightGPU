#pragma once



namespace GLWindow {
	void Resize(int width, int height);
	void MainMenu(int i);
	void InitMenus();
	void CaculateFPS();
	void WindowsUpdate();
	void TimerProc(int id);
	void GlMouseEvent(int button, int state, int x, int y);
	void GlMouseMotion(int x, int y);
	void GLKeyDownEvent(unsigned char key, int /*x*/, int /*y*/);
	void GLKeyUpEvent(unsigned char key, int /*x*/, int /*y*/);
	void InitWindow(int argc, char** argv, unsigned int mode, int x_position, int y_position, int width, int heigth, const char* title);
	void Savepic();
}
