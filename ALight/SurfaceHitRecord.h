#pragma once
#include "float3Extension.h"
#include "ONB.h"
class Material;

class SurfaceHitRecord {
public:
	float t;				// Ray hits at p = Ray.origin() + r*Ray.direction()
	float3 p;				// point of intersection
	float3 texp;			// point of intersection for Texture mapping
	ONB uvw;				// w is the outward normal
	float2 uv;
	Material* mat_ptr;		//

	__device__ SurfaceHitRecord():t(99999),p(make_float3(1,1,1)),texp(make_float3(1,1,1))
	{
		
	}
};

