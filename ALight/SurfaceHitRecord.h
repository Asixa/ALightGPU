#pragma once
class Material;

class SurfaceHitRecord {
public:
	float t;				// Ray hits at p = Ray.origin() + r*Ray.direction()
	float3 p;				// point of intersection
	float3 normal;				// point of intersection
	// float3 texp;			// point of intersection for Texture mapping
	// ONB uvw;				// w is the outward normal
	float2 uv;
	Material* mat_ptr;		//

	__device__ SurfaceHitRecord() : t(99999)
	{
	}

	__device__ SurfaceHitRecord(SurfaceHitRecord* rec)
	{
		t = rec->t;
		p = rec->p;
		normal = rec->normal;
		mat_ptr = rec->mat_ptr;
		uv = rec->uv;
	}
};

