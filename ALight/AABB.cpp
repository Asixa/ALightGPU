#include "AABB.h"
#include <vector_functions.hpp>
#include <cfloat>
#include "float3Extension.h"
#include "Defines.h"

AABB* MakeAABB(Triangle& triangle)
{
	auto aabb=new AABB();
	aabb->min=Minimum(triangle.v1.point, aabb->min);
	aabb->min=Minimum(triangle.v2.point, aabb->min);
	aabb->min=Minimum(triangle.v3.point, aabb->min);

	aabb->max = Maximum(triangle.v1.point, aabb->max);
	aabb->max = Maximum(triangle.v2.point, aabb->max);
	aabb->max = Maximum(triangle.v3.point, aabb->max);

	// aabb->min -= EPSILON;
	// aabb->max += EPSILON;
	return aabb;
}

AABB* MakeAABB(AABB* a, AABB* b)
{
	auto aabb = new AABB();
	aabb->min = Minimum(a->min, aabb->min);
	aabb->min = Minimum(b->min, aabb->min);
	aabb->max = Maximum(a->max, aabb->max);
	aabb->max = Maximum(b->max, aabb->max);
	aabb->min -= EPSILON;
	aabb->max += EPSILON;
	return aabb;
}
