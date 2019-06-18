#include "float3Extension.h"
#include <vector_functions.hpp>
#include "math.h"
__host__ __device__ float3 operator+(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

__host__ __device__ float3 operator+(const float3& v, const float& t)
{
	return make_float3(v.x + t, v.y + t, v.z + t);
}

__host__ __device__ float3 operator-(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

__host__ __device__ float3 operator*(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}

__host__ __device__ float3 operator*(float t, const float3& v)
{
	return make_float3(v.x * t, v.y * t, v.z *t);
}

__host__ __device__ float3 operator*(const float3& v, float t)
{
	return make_float3(v.x * t, v.y * t, v.z * t);
}

__host__ __device__ float3 operator/(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}

__host__ __device__ float3 operator/(float3 v, float t)
{
	return make_float3(v.x / t, v.y / t, v.z / t);
}

__host__ __device__ float Dot(const float3& v1, const float3& v2)
{
	return v1.x* v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ float3 Cross(const float3& v1, const float3& v2)
{
	return make_float3(v1.y * v2.z - v1.z * v2.y, (-(v1.x * v2.z - v1.z * v2.x)),
		(v1.x * v2.y - v1.y * v2.x));
}

__host__ __device__ float3 Cross2(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.y * rhs.z - lhs.z * rhs.y,
		lhs.z * rhs.x - lhs.x * rhs.z,
		lhs.x * rhs.y - lhs.y * rhs.x);
}

__host__ __device__ float Length(float3 f)
{
	return sqrt(f.x * f.x + f.y * f.y + f.z * f.z);
}

__host__ __device__ float SquaredLength(float3 f)
{
	return f.x * f.x + f.y * f.y + f.z * f.z;
}

__host__ __device__ bool IsZero(float3 f)
{
	return f.x == 0 && f.y == 0 && f.z == 0;
}

__host__ __device__ float3 UnitVector(float3 v)
{
	return v / Length(v);
}



__host__ __device__ float Distance(float3 a, float3 b)
{
	return Length(a - b);
}

__host__ __device__ void MakeUnitVector(float3* f)
{
	const float k = 1.0 / sqrt(f->x * f->x + f->y * f->y + f->z * f->z);
	f->x *= k; f->y *= k; f->z *= k;
}

__host__ __device__ float3 Reflect(float3 vin, float3 normal)
{
	return vin - 2 * Dot(vin, normal) * normal;
}

__host__ __device__ float3 Min(float3 a, float3 b)
{
	if (Length(a) < Length(b))return a;
	else return b;
}

__host__ __device__ float3 operator-(float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ float3 operator-(const float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

void Set(float3& f, float a, float b, float c)
{
	f.x = a;
	f.y = b;
	f.z = c;
}

void Set(float3& f, float a)
{
	f.x = a;
	f.y = a;
	f.z = a;
}
