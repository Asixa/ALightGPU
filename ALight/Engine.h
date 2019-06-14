#pragma once
#include "RayTracer.h"
#include "Camera.h"

class Engine
{
	float camera_w=M_PI/2,camera_y=2,camera_r=5;
	static Engine* instance;
public:
	Engine() {}
	~Engine(){};
	RayTracer* RayTracer;
	Camera* camera;


	void Init();
	void Update();
	void OnMouseMove(int x, int y);
	void OnMouseScroll(int a);

	static Engine* Instance()
	{
		if (!instance)instance = new Engine();
		return instance;
	}
};
