#pragma once
#include "curand_kernel.h"
#include <GL/glew.h>
#include "vec3.h"
#include "Material.h"


class DeviceManager;
class RayTracer
{
	// GLbyte* data;
	void SetPixel(const int x, const int y, Vec3* c) const;
public:
	RayTracer();
	~RayTracer();

	Material* Materials;

	bool GPU;
	DeviceManager* device_manager;
	GLbyte* Data;
	int Sampled = 0;
	int Width, Height;
	explicit RayTracer(bool GPU);
	void Init(GLbyte* data,int w,int h);
	void Render() const;
};
