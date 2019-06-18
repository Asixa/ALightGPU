#pragma once
#include <crt/host_defines.h>
#include <driver_functions.h>
#include <vector_functions.hpp>
#include <math.h>
// float3 Createfloat3(float x, float y, float z);
__host__ __device__  float3 operator+(const float3& v1, const float3& v2);
__host__ __device__  float3 operator+(const float3& v, const float& t);
__host__ __device__  float3 operator-(const float3& v1, const float3& v2);
__host__ __device__  float3 operator*(const float3& v1, const float3& v2);
__host__ __device__  float3 operator*(float t, const float3& v);
__host__ __device__  float3 operator*(const float3& v, float t);
__host__ __device__  float3 operator/(const float3& v1, const float3& v2);
__host__ __device__  float3 operator/(float3 v, float t);


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

 __host__ __device__ float3 operator-(float3& a);
 __host__ __device__ float3 operator-(const float3& a);


 __host__ __device__ void Set(float3& f, float a, float b, float c);
 __host__ __device__ void Set(float3& f, float a);

 inline __host__ __device__ void operator+=(float3& a, float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}
 inline __host__ __device__ void operator+=(float3& a, float3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
 inline  __host__ __device__ void operator-=(float3& a, float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}
 inline  __host__ __device__ void operator-=(float3& a, float3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
 inline __host__ __device__ void operator*=(float3& a, float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}
 inline  __host__ __device__ void operator*=(float3& a, float3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
 inline __host__ __device__ void operator/=(float3& a, float3 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}
 inline __host__ __device__ void operator/=(float3& a, float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}



 __host__ __device__ __inline__
	 float3 Maximum(const float3& a, const float3& b) {
	 return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
 }

 __host__ __device__ __inline__
	 float3 Minimum(const float3& a, const float3& b) {
	 return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
 }

 __host__ __device__ __inline__
	 float Get(const float3& a, const int i) {
	 return i == 0 ? a.x : (i == 1 ? a.y : a.z);
 }

	