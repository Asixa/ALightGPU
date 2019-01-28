#pragma once
#include "vec3.h"
#include "Material.h"

class Sphere : public Hitable {
public:
	__device__ __host__ Sphere() { type = Instance::SPHERE; }
	//Sphere(Vec3 cen, float r, Material *m) : center(cen), radius(r), mat_ptr(m) {};
	__device__ __host__ Sphere(Vec3 cen, float r, int m) : center(cen), radius(r), mat_id(m) { type = Instance::SPHERE; };

	__device__  Sphere(Hitable* data)
	{
		id = data->id;
		type = Instance::SPHERE;
		auto mirror = static_cast<Sphere*>(data);
		center = mirror->center;
		radius = mirror->radius;
		mat_id = mirror->mat_id;
		//delete data;
	};
	__device__ bool Hit(const Ray& r, float tmin, float tmax, HitRecord& rec,DeviceData* data) const override;
	__host__ bool BoundingBox(float t0, float t1, AABB& box) const override;
	int Size() const override;
	__device__ int Debug() const override;
	Hitable* GPUPointer() override;
	int count() override;
	//__device__ __host__ int type() override;

	Vec3 center;
	float radius;
	int mat_id;
	//Material *mat_ptr;
};


bool Sphere::BoundingBox(float t0, float t1, AABB& box) const {
	box = AABB(center - Vec3(radius, radius, radius), center + Vec3(radius, radius, radius));
	return true;
}

inline int Sphere::Size() const
{
	return sizeof(Sphere);
}

__device__ inline int Sphere::Debug() const
{
	return 666;
}

inline Hitable* Sphere::GPUPointer()
{
	Hitable* pointer;
	cudaMalloc(&pointer, sizeof(Sphere));
	cudaMemcpy(pointer, this, sizeof(Sphere), cudaMemcpyHostToDevice);
	return pointer;
}

inline int Sphere::count()
{
	return 0;
}

// __device__ __host__ inline int Sphere::type()
// {
// 	return 1;
// }


bool Sphere::Hit(const Ray& r, float t_min, float t_max, HitRecord& rec, DeviceData* data) const {
	Vec3 oc = r.Origin() - center;
	float a = dot(r.Direction(), r.Direction());
	float b = dot(oc, r.Direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(b*b - a * c)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.PointAtParameter(rec.t);
			GetSphereUv((rec.p - center) / radius, rec.u, rec.v);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = &data->materials[mat_id];
			return true;
		}
		temp = (-b + sqrt(b*b - a * c)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.PointAtParameter(rec.t);
			GetSphereUv((rec.p - center) / radius, rec.u, rec.v);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = &data->materials[mat_id];
			return true;
		}
	}
	return false;
	// auto oc = r.Origin() - center;
	// float a = dot(r.Direction(), r.Direction());
	// float b = 2*dot(oc, r.Direction());
	// auto c = dot(oc, oc) - radius * radius;
	// const auto discriminant = b * b -4* a * c;
	//
	// if (discriminant > 0)
	// {
	// 	float temp = (-b - sqrt(discriminant)) / a * 0.5f;
	// 	if (temp<t_max&&temp>t_min)
	// 	{
	// 		rec.t = temp;
	// 		rec.p = r.PointAtParameter(rec.t);
	// 		rec.normal = (rec.p - center) / radius;
	// 		rec.mat_ptr = &data->materials[mat_id];
	// 		GetSphereUv((rec.p - center) / radius, rec.u, rec.v);
	// 		return true;
	// 	}
	// 	temp = (-b + sqrt(discriminant)) / a * 0.5f;
	// 	if (temp<t_max&&temp>t_min)
	// 	{
	// 		rec.t = temp;
	// 		rec.p = r.PointAtParameter(rec.t);
	// 		rec.normal = (rec.p - center) / radius;
	// 		rec.mat_ptr = &data->materials[mat_id];
	// 		GetSphereUv((rec.p - center) / radius, rec.u, rec.v);
	// 		return true;
	// 	}
	// }
	// //printf("%d\n", -1);
	// return false;
}
