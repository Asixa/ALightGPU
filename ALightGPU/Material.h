#pragma once
#include "vec3.h"
#include "Hitable.h"


class Material;

struct HitRecord
{
	float t;
	Vec3 p;
	Vec3 normal;
	Material *mat_ptr;
};


__device__ float Schlick(float cosine, float ref_idx) {
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0 * r0;
	return r0 + (1 - r0)*pow((1 - cosine), 5);
}

__device__ inline bool Refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted) {
	Vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0 - ni_over_nt * ni_over_nt*(1 - dt * dt);
	if (discriminant > 0)
	{
		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	else
		return false;
}

__device__ inline Vec3 Reflect(const Vec3& v, const Vec3& n)
{
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
	bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, Vec3 random_in_unit_sphere,
	             float randomnumber);
};

__device__ inline bool Material::scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered,
                                         Vec3 random_in_unit_sphere, float randomnumber)
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
			if (dot(r_in.Direction(), rec.normal) > 0)
			{
				outward_normal = -rec.normal;
				ni_over_nt = ref_idx;
				//         cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
				cosine = dot(r_in.Direction(), rec.normal) / r_in.Direction().length();
				cosine = sqrt(1 - ref_idx * ref_idx*(1 - cosine * cosine));
			}
			else
			{
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
