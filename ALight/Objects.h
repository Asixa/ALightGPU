#pragma once
#include "float3Extension.h"

struct Sphere
{
    float3 position;
    float radius;
    float3 albedo;
    float3 specular;
	__device__ Sphere(float3 p,float r,float3 a,float3  s):position(p),radius(r),albedo(a),specular(s)
	{
		
	}
};

