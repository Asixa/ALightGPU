#pragma once
#include "vec3.h"
#include "ray.h"
#include <valarray>
#define MATERIAL_PARAMTER_COUNT 6
#define HITABLE_PARAMTER_COUNT 5

#define LAMBERTIAN 1
#define METAL 2
#define DIELECTIRC 3

class AABB;
class Material;
struct HitRecord
{
	float t;
	Vec3 p;
	Vec3 normal;
	Material *mat_ptr;
};


class GPUHitable
{
	
public:
	float data[HITABLE_PARAMTER_COUNT];
	GPUHitable(float d[HITABLE_PARAMTER_COUNT])
	{
		memcpy(data, d, HITABLE_PARAMTER_COUNT * sizeof(float));
	};
	virtual ~GPUHitable() = default;
	bool Hit(const Ray& r, float tmin, float tmax, HitRecord& rec, Material* materials);
	bool bounding_box(float t0, float t1, AABB& box);
};







//


__device__ float Schlick(float cosine, float ref_idx) {
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0 * r0;
	return r0 + (1 - r0)*pow((1 - cosine), 5);
}

__device__ inline bool Refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted) {
	Vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0 - ni_over_nt * ni_over_nt*(1 - dt * dt);
	if (discriminant > 0) {
		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	else
		return false;
}

__device__ inline Vec3 Reflect(const Vec3& v, const Vec3& n) {
	return v - 2 * dot(v, n)*n;
}

class Material {
public:
	float data[MATERIAL_PARAMTER_COUNT];
	int type;
	Material(int t, float d[MATERIAL_PARAMTER_COUNT])
	{
		type = t;
		memcpy(data, d, MATERIAL_PARAMTER_COUNT * sizeof(float));
	}
	bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered,Vec3 random_in_unit_sphere, float randomnumber);
};

__device__ inline bool Material::scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, Vec3 random_in_unit_sphere, float randomnumber)
{
	switch (type)
	{
	case 1:
	{
		auto albedo = Vec3(data[0], data[1], data[2]);
		Vec3 target = rec.p + rec.normal + random_in_unit_sphere;
		scattered = Ray(rec.p, target - rec.p);
		attenuation = albedo;
		return true;
	}
	case 2:
	{
		auto albedo = Vec3(data[0], data[1], data[2]);
		auto fuzz = data[3];

		Vec3 reflected = Reflect(unit_vector(r_in.Direction()), rec.normal);
		scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere);
		attenuation = albedo;
		return (dot(scattered.Direction(), rec.normal) > 0);
	}
	case 3:
	{

		auto ref_idx = data[0];

		Vec3 outward_normal;
		Vec3 reflected = Reflect(r_in.Direction(), rec.normal);
		float ni_over_nt;
		attenuation = Vec3(1.0, 1.0, 1.0);
		Vec3 refracted;
		float reflect_prob;
		float cosine;
		if (dot(r_in.Direction(), rec.normal) > 0) {
			outward_normal = -rec.normal;
			ni_over_nt = ref_idx;
			//         cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
			cosine = dot(r_in.Direction(), rec.normal) / r_in.Direction().length();
			cosine = sqrt(1 - ref_idx * ref_idx*(1 - cosine * cosine));
		}
		else {
			outward_normal = rec.normal;
			ni_over_nt = 1.0 / ref_idx;
			cosine = -dot(r_in.Direction(), rec.normal) / r_in.Direction().length();
		}
		if (Refract(r_in.Direction(), outward_normal, ni_over_nt, refracted))
			reflect_prob = Schlick(cosine, ref_idx);
		else
			reflect_prob = 1.0;
		if (randomnumber < reflect_prob)
			scattered = Ray(rec.p, reflected);
		else
			scattered = Ray(rec.p, refracted);
		return true;
	}
	}

}

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

inline float ffmin(float a, float b) { return a < b ? a : b; }
inline float ffmax(float a, float b) { return a > b ? a : b; }


class AABB {
public:
	AABB() {}
	AABB(const Vec3& a, const Vec3& b) { _min = a; _max = b; }

	bool Hit(const Ray& r, float tmin, float tmax) const {
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