#pragma once
#include "Hitable.h"
#include "Vertex.h"

#define EPSILON  1e-4f
class Triangle:public Hitable
{
public:
	Vertex v0, v1, v2;
	Vec3 GNormal;
	int mat_id;

	__device__ __host__ Triangle() { type = Instance::TRIANGLE; }

	__device__ __host__ Triangle(Vertex a, Vertex b, Vertex c, int mat):v0(a),v1(b),v2(c),mat_id(mat)
	{
		type = Instance::TRIANGLE;
		GNormal = (a.normal + b.normal + c.normal) / 3;
	}
	__device__ Triangle(Hitable* data)
	{
		type = Instance::TRIANGLE;
		id = data->id;
		const auto mirror = static_cast<Triangle*>(data);
		v0 = mirror->v0;
		v1 = mirror->v1;
		v2 = mirror->v2;
		GNormal = mirror->GNormal;
		mat_id = mirror->mat_id;
	}

	__device__ Vec2 GetUV(Vec3 p, Vec3& normal) const;
	__device__ bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec, DeviceData* data) const override;
	__host__ bool BoundingBox(float t0, float t1, AABB& box) const override;
	int Size() const override;
	__device__ int Debug() const override;
	__device__ bool Intersects(Vec3 ray_origin, Vec3 ray_dir, Vec3& point, float& T) const;
};

inline Vec2 Triangle::GetUV(Vec3 p, Vec3& normal) const
{
	auto f1 = v0.point - p;
	auto f2 = v1.point - p;
	auto f3 = v2.point - p;
	//计算面积和因子（参数顺序无关紧要）：
	auto a = cross(v0.point - v1.point, v0.point - v2.point).Length(); // 主三角形面积 a
	auto a1 = cross(f2, f3).Length() / a; // p1 三角形面积 / a
	auto a2 = cross(f3, f1).Length() / a; // p2 三角形面积 / a 
	auto a3 = cross(f1, f2).Length() / a; // p3 三角形面积 / a
	// 找到对应于点f的uv（uv1 / uv2 / uv3与p1 / p2 / p3相关）：
	auto uv = v0.uv * a1 + v1.uv * a2 + v2.uv * a3;
	// 找到对应于点f的法线（法线1 / 法线2 / 法线3与p1 / p2 / p3相关）：
	normal = v0.normal * a1 + v1.normal * a2 + v2.normal * a3;
	return uv;
}

inline bool Triangle::Hit(const Ray& r, float t_min, float t_max, HitRecord& rec, DeviceData* data)const
{
	//printf("Check");
	if (data->materials[mat_id].BackCulling&&dot(GNormal, r.Direction()) >= 0)return false;
	Vec3 p;
	float t;
	if (!Intersects(r.Origin(), unit_vector(r.Direction()), p,t)) return false;
	rec.t = Distance(r.Origin(),p)/2;
	rec.p = p;
	rec.mat_ptr = &data->materials[mat_id];
	const auto uvw = GetUV(rec.p,p);
	rec.normal = p;
	rec.u = uvw.X;
	rec.v = uvw.Y;
	// rec.bitangent = v0.bitangent;
	// rec.tangent = v0.tangent;
	return true;
}

inline bool Triangle::BoundingBox(float t0, float t1, AABB& box) const
{
	const auto bl = Vec3(
		min(min(v0.point[0], v1.point[0]), v2.point[0]),
		min(min(v0.point[1], v1.point[1]), v2.point[1]),
		min(min(v0.point[2], v1.point[2]), v2.point[2]));
	const auto tr = Vec3(
		max(max(v0.point[0], v1.point[0]), v2.point[0]),
		max(max(v0.point[1], v1.point[1]), v2.point[1]),
		max(max(v0.point[2], v1.point[2]), v2.point[2]));
	box = AABB(bl - Vec3(0.1f, 0.1f, 0.1f), tr + Vec3(0.1f, 0.1f, 0.1f));
	return true;
}

inline int Triangle::Size() const
{
	return sizeof(Triangle);
}

inline int Triangle::Debug() const
{
	return 999;
}

inline bool Triangle::Intersects(Vec3 ray_origin, Vec3 ray_dir, Vec3& point,float& T) const
{
	const auto edge1 = v1.point - v0.point;
	const auto edge2 = v2.point - v0.point;
	const auto h = cross(ray_dir, edge2);
	const auto a = dot(edge1, h);
	if (a > -EPSILON && a < EPSILON)return false;
	const auto f = 1 / a;
	const auto s = ray_origin - v0.point;
	const auto u = f * (dot(s, h));
	if (u < 0.0 || u > 1.0)return false;
	const auto q = cross(s, edge1);
	const auto v = f *dot(ray_dir, q);
	if (v < 0.0 || u + v > 1.0)return false;
	const auto t = f * dot(edge2, q);
	if (!(t > EPSILON)) return false;
	point = ray_origin + ray_dir * t;
	T = t;
	return true;
}
