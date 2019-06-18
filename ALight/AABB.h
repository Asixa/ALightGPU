#pragma once
#include <vector_types.h>
#include "Triangle.h"
#include <cfloat>
#include <vector_functions.hpp>
struct AABB;
AABB* MakeAABB(Triangle& triangle);
AABB* MakeAABB(AABB* a, AABB* b);
struct AABB
{
	float3 min, max;
	AABB()
	{
		min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	}
};
