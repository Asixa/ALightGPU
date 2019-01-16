#pragma once
#include "vec3.h"

class Ray
{
public:
	Ray() = default;
	Ray(const Vec3& a, const Vec3& b) { A = a; B = b; }
	Vec3 Origin() const { return A; }
	Vec3 Direction() const { return B; }
	Vec3 PointAtParameter(const float t) const { return A + t * B; }
	Vec3 A;
	Vec3 B;
};