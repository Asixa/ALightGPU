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
	Material* materials;
	int material_count;
	bool GPU;
	DeviceManager* device_manager;
	GLbyte* data;
	int sampled = 0;
	int width, height;
	cudaTextureObject_t textlist[1];
	bool IPR_Quick=false, IPR_reset_once=false;
	void ReSetIPR();
	explicit RayTracer(bool GPU);

	void Init(GLbyte* data,int w,int h);
	void Render() const;
};
