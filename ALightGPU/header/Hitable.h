#pragma once
#include "vec3.h"
#include "ray.h"
struct HitRecord
{
	float t;
	Vec3 p;
	Vec3 normal;
};

class Hitable
{
public:
	virtual ~Hitable() = default;
	virtual bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec)const = 0;
};
