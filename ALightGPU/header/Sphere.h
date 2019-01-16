#pragma once
#include "vec3.h"
#include "ray.h"
#include "Hitable.h"

class Sphere :public Hitable
{
public:
	Sphere() = default;
	Sphere(const Vec3 cent, float r) :center(cent), radius(r) {};
	virtual bool Hit(const Ray& r, float tmin, float tmax, HitRecord& rec)const;
	Vec3 center;
	float radius;
};

inline bool Sphere::Hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const
{
	Vec3 oc = r.Origin() - center;
	float a = dot(r.Direction(), r.Direction());
	float b =dot(oc, r.Direction());
	auto c = dot(oc, oc) - radius * radius;
	const auto discriminant = b * b - a*c;

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
