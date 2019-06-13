#pragma once
#include <crt/host_defines.h>
#include <driver_functions.h>

namespace Float2 {
	__host__ __device__ float Dot(const float2& v1, const float2& v2);
	// __host__ __device__ float3 Cross(const float3& v1, const float3& v2);
	// __host__ __device__ float3 Cross2(const float3& lhs, const float3& rhs);
	// __host__ __device__ float Length(float3 f);
	// __host__ __device__ float SquaredLength(float3 f);
	// __host__ __device__ bool IsZero(float3 f);
	// __host__ __device__ float3 UnitVector(float3 v);
	// __host__ __device__ float Distance(float3 a, float3 b);
	// __host__ __device__ void MakeUnitVector(float3* b);
	// __host__ __device__ float3 Reflect(float3 vin, float3 normal);
	// __host__ __device__ float3 Min(float3 a, float3 b);
	__host__ __device__ void Scramble(float2& v);

}


