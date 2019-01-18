#pragma once
#include "vec3.h"
#include "root.h"

class Ray
{
public:
	__device__ Ray() = default;
	__device__ Ray(const Vec3& a, const Vec3& b) { A = a; B = b; }
	__device__ Vec3 Origin() const { return A; }
	__device__ Vec3 Direction() const { return B; }
	__device__ Vec3 PointAtParameter(const float t) const { return A + t * B; }
	Vec3 A;
	Vec3 B;
};
