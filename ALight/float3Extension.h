#pragma once
#include <crt/host_defines.h>
#include <driver_functions.h>

// float3 Createfloat3(float x, float y, float z);
__host__ __device__  float3 operator+(const float3& v1, const float3& v2);
__host__ __device__  float3 operator+(const float3& v, const float& t);
__host__ __device__  float3 operator-(const float3& v1, const float3& v2);
__host__ __device__  float3 operator*(const float3& v1, const float3& v2);
__host__ __device__  float3 operator*(float t, const float3& v);
__host__ __device__  float3 operator*(const float3& v, float t);
__host__ __device__  float3 operator/(const float3& v1, const float3& v2);
__host__ __device__  float3 operator/(float3 v, float t);
__host__ __device__  bool operator==(float3 a, float3 b);

__host__ __device__  float3 &operator+=(float3 x,const float3& v);
__host__ __device__  float3 &operator*=(float3 x,const float3& v);
__host__ __device__  float3 &operator*=(float x,const float3& v);
__host__ __device__  float3 &operator/=(float3 x,const float3& v);
__host__ __device__  float3 &operator/=(float x ,const float3& v);
__host__ __device__  float3 &operator-=(float3 x,const float3& v);
__host__ __device__  float3 &operator-(const float3& v);

namespace Float3 {
	__host__ __device__ float Dot(const float3& v1, const float3& v2);
	__host__ __device__ float3 Cross(const float3& v1, const float3& v2);
	__host__ __device__ float3 Cross2(const float3& lhs, const float3& rhs);
	__host__ __device__ float Length(float3 f);
	__host__ __device__ float SquaredLength(float3 f);
	__host__ __device__ bool IsZero(float3 f);
	__host__ __device__ float3 UnitVector(float3 v);
	__host__ __device__ float Distance(float3 a, float3 b);
	__host__ __device__ void MakeUnitVector(float3* b);
	__host__ __device__ float3 Reflect(float3 vin, float3 normal);
	__host__ __device__ float3 Min(float3 a, float3 b);

}