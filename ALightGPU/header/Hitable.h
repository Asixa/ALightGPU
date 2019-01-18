#pragma once
#include "vec3.h"
#include "ray.h"
#include <valarray>

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

class GPUHitable
{
	
public:
	float data[4];
	GPUHitable(float d[4])
	{
		memcpy(data, d, 4 * sizeof(float));
		//printf("复合球体数据%f,%f,%f,%f\n", data[0], data[1], data[2], data[3]);
	};
	virtual ~GPUHitable() = default;
	bool Hit(const Ray& r, float tmin, float tmax, HitRecord& rec);
};

__device__ inline bool GPUHitable::Hit(const Ray& r, float tmin, float tmax, HitRecord& rec)
{
		auto center = Vec3(data[0],data[1],data[2]);
		auto radius = data[3];

		//printf("检测数据： center: %f,%f,%f  radius %f\n", center.x(), center.y(), center.z(), radius);

		Vec3 oc = r.Origin() - center;
		float a = dot(r.Direction(), r.Direction());
		float b = dot(oc, r.Direction());
		auto c = dot(oc, oc) - radius * radius;
		const auto discriminant = b * b - a * c;

		if (discriminant > 0)
		{
			float temp = (-b - sqrt(b*b - a * c)) / a;
			if (temp<tmax&&temp>tmin)
			{
				rec.t = temp;
				rec.p = r.PointAtParameter(rec.t);
				rec.normal = (rec.p - center) / radius;
				return true;
			}
			temp = (-b + sqrt(b*b - a * c)) / a;
			if (temp<tmax&&temp>tmin)
			{
				rec.t = temp;
				rec.p = r.PointAtParameter(rec.t);
				rec.normal = (rec.p - center) / radius;
				return true;
			}
		}
		return false;
	
}
