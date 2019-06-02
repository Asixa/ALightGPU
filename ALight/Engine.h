#pragma once
#include "RayTracer.h"
#include "Camera.h"

class Engine
{
	
	static Engine* instance;
public:
	Engine() {}
	~Engine();
	RayTracer* RayTracer;
	Camera* camera;
	void Init();
	void Update();

	static Engine* Instance()
	{
		if (!instance)instance = new Engine();
		return instance;
	}
};
