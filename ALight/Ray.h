#pragma once
#include <device_launch_parameters.h>

__host__ __device__ struct Ray
{
	float3 origin;
	float3 direction;
	float3 energy;

};



