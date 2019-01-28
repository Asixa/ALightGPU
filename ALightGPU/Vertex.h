#pragma once
#include "vec3.h"

struct Vertex
{
public:
	Vec3 point, normal, tangent, bitangent;
	Vec2 uv;
	__device__ __host__ Vertex(){}
	__device__ __host__ Vertex(const Vec3 p, const Vec3 n, const float u, const float v)
	:point(p), normal(n), uv(Vec2(u, v))
	{

	}

};
