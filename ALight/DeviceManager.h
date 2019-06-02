#pragma once
#include "RayTracer.h"
#include "Camera.h"

class DeviceManager
{
	curandState* rng_states;
	float* devicde_float_data;
	GLbyte* devicde_byte_data;
	float* host_float_data;
	RayTracer* ray_tracer;
	dim3 grid;
	dim3 block;

	Camera* d_camera;
public:
	DeviceManager();
	~DeviceManager();
	void PrintDeviceInfo();
	void Init(RayTracer* ray_tracer);
	void Run();
};
