#pragma once
#include "vec3.h"
#include "ray.h"
#include "root.h"

inline Vec3 RandomInUnitDisk() {
	Vec3 p;
	do {
		p = 2.0*Vec3(drand48(), drand48(), 0) - Vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0);
	return p;
}

inline Vec3 RandomInUnitSphere() {
	Vec3 p;
	do {
		p = 2.0*Vec3(drand48(), drand48(), drand48()) - Vec3(1, 1, 1);
	} while (p.squared_length() >= 1.0);
	return p;
}
class Camera
{
public:
	Camera()
	{
		LowerLeftCorner=Vec3(-2.0, -2.0, -1.0), 
		Horizontal = Vec3(4, 0, 0), 
		Vertical = Vec3(0, 4, 0), 
		Origin = Vec3(0, 0, 0);
	}
	Ray GetRay(float u,float v) const
	{
		return Ray(Origin, LowerLeftCorner + u * Horizontal + v * Vertical);
	}

	Vec3 Origin, LowerLeftCorner, Horizontal, Vertical;
};
