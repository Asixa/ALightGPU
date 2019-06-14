#pragma once
#include "MathHelper.h"
__host__ __device__ struct Ray
{
	float3 origin;
	float3 direction;
	// float3 energy;
	float3 PointAtParameter(float t) const { return origin +  direction* t; }

	__device__ Ray(float3 origin, float3 direction):origin(origin),direction(direction)
	{
		
	}

	__device__ Ray()
	{

	}
};



