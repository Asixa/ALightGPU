#pragma once
#include <device_launch_parameters.h>
#include "Material.h"

__host__ __device__ struct RayHit
{
	float3 position;
	float distance;
	float3 normal;
	float3 albedo;
	float3 specular;
	float smoothness;
	float3 emission;
};

