#pragma once
#include <cuda_runtime.h>
#include <cstring>
#include "MathHelper.h"
// #include <crt/host_defines.h>
// #include <driver_functions.h>
// #include <cstring>


struct Ray;
class SurfaceHitRecord;
class RTDeviceData;
#define MATERIAL_PARAMTER_COUNT 6

__device__ float Schlick(float cosine, float ref_idx);
__device__ bool Refract(const float3& v, const float3& n, float ni_over_nt, float3& refracted);

enum MaterialType
{
	lambertian,
	metal,
	dielectirc
};

class Material {
public:
	bool BackCulling = true;
	float data[MATERIAL_PARAMTER_COUNT];
	//int type;
	MaterialType Type;

	__device__ Material() {}
	__device__ Material(MaterialType t, float d[MATERIAL_PARAMTER_COUNT])
	{
		Type = t;
		memcpy(data, d, MATERIAL_PARAMTER_COUNT * sizeof(float));
	}
	__device__ bool scatter(const Ray& r_in, const SurfaceHitRecord& rec, float3& attenuation, Ray& scattered, float3 random_in_unit_sphere, const RTDeviceData& data);
};

