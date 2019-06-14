#pragma once
class Material;

class SurfaceHitRecord {
public:
	float t;			
	float3 p;			
	float3 normal;			
	float2 uv;
	Material* mat_ptr;		

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

