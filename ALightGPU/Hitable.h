#pragma once
#include "vec3.h"
#include "ray.h"
#include <valarray>
#include "Material.h"
#include "AABB.h"


class GPUHitable
{
	
public:
	float data[HITABLE_PARAMTER_COUNT];
	GPUHitable(float d[HITABLE_PARAMTER_COUNT])
	{
		memcpy(data, d, HITABLE_PARAMTER_COUNT * sizeof(float));
	};
	virtual ~GPUHitable() = default;
	__device__  bool Hit(const Ray& r, float tmin, float tmax, HitRecord& rec, Material* materials);
	bool bounding_box(float t0, float t1, AABB& box);
};

__device__ inline bool GPUHitable::Hit(const Ray& r, float tmin, float tmax, HitRecord& rec, Material* materials)
{
	//SetData
	auto center = Vec3(data[0], data[1], data[2]);
	auto radius = data[3];
	auto material = int(data[4]);

	auto oc = r.Origin() - center;
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
			rec.mat_ptr = &materials[material];
			return true;
		}
		temp = (-b + sqrt(b*b - a * c)) / a;
		if (temp<tmax&&temp>tmin)
		{
			rec.t = temp;
			rec.p = r.PointAtParameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = &materials[material];
			return true;
		}
	}
	return false;
}

inline bool GPUHitable::bounding_box(float t0, float t1, AABB& box)
{
}
namespace Instance
{
	const int BVH = 2,SPHERE=1;
}
class Hitable {
public:
	virtual ~Hitable() = default;
	
	__device__ virtual bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec,Material* materials, Hitable** d_world) const = 0;
	__host__  virtual bool BoundingBox(float t0, float t1, AABB& box) const = 0;
	__host__  virtual int Size() const = 0;
	__device__  virtual int Debug() const = 0;
	__host__ virtual Hitable* GPUPointer() { return nullptr; }
	__host__ virtual int count() { return 0; }
	__host__ virtual void SetChildId() { }
	//__device__ __host__  virtual int type() { return 0;}
	int id; int type;
};
