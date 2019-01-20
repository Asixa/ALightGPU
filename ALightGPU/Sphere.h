#pragma once
#include "vec3.h"
#include "Material.h"

class Sphere : public Hitable {
public:
	Sphere() {}
	//Sphere(Vec3 cen, float r, Material *m) : center(cen), radius(r), mat_ptr(m) {};
	Sphere(Vec3 cen, float r, int m) : center(cen), radius(r), mat_id(m) {};
	bool Hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const override;
	bool BoundingBox(float t0, float t1, AABB& box) const override;
	Vec3 center;
	float radius;
	int mat_id;
	//Material *mat_ptr;
};


bool Sphere::BoundingBox(float t0, float t1, AABB& box) const {
	box = AABB(center - Vec3(radius, radius, radius), center + Vec3(radius, radius, radius));
	return true;
}

bool Sphere::Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
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
			//get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);
			rec.normal = (rec.p - center) / radius;
			//rec.mat_ptr = mat_ptr;
			return true;
		}
		temp = (-b + sqrt(b*b - a * c)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.PointAtParameter(rec.t);
			//get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);
			rec.normal = (rec.p - center) / radius;
			//rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}