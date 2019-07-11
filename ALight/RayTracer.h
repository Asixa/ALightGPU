#pragma once
#include "curand_kernel.h"
#include <GL/glew.h>
#include "Material.h"


class DeviceManager;
class RayTracer
{
	// GLbyte* data;

public:
	RayTracer();
	~RayTracer();

	bool GPU,Done=false;
	DeviceManager* device_manager;
	GLbyte* data;
	int sampled = 0;
	int width, height;
	
	bool thingsChanged;
	bool IPR_Quick=false, IPR_reset_once=false;
	void ReSetIPR();
	explicit RayTracer(bool GPU);

	void Init(GLbyte* data,int w,int h);
	void Render();
};
