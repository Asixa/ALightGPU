#pragma once
#include "vec3.h"
#include "ray.h"

__device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
__device__ inline float ffmax(float a, float b) { return a > b ? a : b; }

class AABB {
public:
	__host__ __device__ AABB(){}
	__host__ AABB(const Vec3& a, const Vec3& b) { _min = a; _max = b; }

	__device__ bool Hit(const Ray& r, float tmin, float tmax) const {
		for (auto a = 0; a < 3; a++) {
			const float t0 = ffmin((_min[a] - r.Origin()[a]) / r.Direction()[a],
				(_max[a] - r.Origin()[a]) / r.Direction()[a]);
			const float t1 = ffmax((_min[a] - r.Origin()[a]) / r.Direction()[a],
				(_max[a] - r.Origin()[a]) / r.Direction()[a]);
			tmin = ffmax(t0, tmin);
			tmax = ffmin(t1, tmax);
			if (tmax <= tmin)
				return false;
		}
		return true;
	}

	Vec3 _min;
	Vec3 _max;
};

inline AABB SurroundingBox(AABB box0, AABB box1) {
	const Vec3 _small(fmin(box0._min.x(), box1._min.x()),
		fmin(box0._min.y(), box1._min.y()),
		fmin(box0._min.z(), box1._min.z()));
	const Vec3 big(fmax(box0._max.x(), box1._max.x()),
		fmax(box0._max.y(), box1._max.y()),
		fmax(box0._max.z(), box1._max.z()));
	return AABB(_small, big);
}