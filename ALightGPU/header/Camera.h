#pragma once
#include "vec3.h"
#include "ray.h"
#include "root.h"
#include <curand_kernel.h>


class Camera
{
public:
	__device__ Camera()
	{
		// LowerLeftCorner=Vec3(-2.0, -2.0, -1.0), 
		// Horizontal = Vec3(4, 0, 0), 
		// Vertical = Vec3(0, 4, 0), 
		// Origin = Vec3(0, 0, 0);
	}
	__device__ Ray GetRay(float u,float v) const
	{
		return Ray(Origin, LowerLeftCorner + u * Horizontal + v * Vertical);
	}

	Vec3 Origin = Vec3(0, 0, 0), 
	LowerLeftCorner = Vec3(-2.0, -2.0, -1.0), 
	Horizontal = Vec3(4, 0, 0), 
	Vertical = Vec3(0, 4, 0);
};
