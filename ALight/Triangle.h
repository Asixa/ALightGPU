#pragma once
#include <vector_types.h>
#include "float3Extension.h"

struct Vertice
{
	float3 point, normal;
	float2 uv;
	Vertice(const float3 p, const float3 n, const float2 u) :point(p), normal(n), uv(u){}
	Vertice() :point(make_float3(0,0,0)), normal(make_float3(0, 0, 0)), uv(make_float2(0, 0)){}
};
struct Triangle
{
	Vertice v1, v2, v3; int mat;
	Triangle(Vertice a,Vertice b,Vertice c,int m):v1(a),v2(b),v3(c),mat(m){}
	Triangle();

};
