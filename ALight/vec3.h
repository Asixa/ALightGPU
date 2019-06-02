#pragma once
#include <cuda_runtime_api.h>
// ReSharper disable CppInconsistentNaming
class Vec3 {

public:
	float e[3];
	__host__ __device__ Vec3();
	__host__ __device__ Vec3(float e0, float e1, float e2);

	__host__ __device__  float x() const;
	__host__ __device__  float y() const;
	__host__ __device__  float z() const;
	__host__ __device__  float r() const;
	__host__ __device__  float g() const;
	__host__ __device__  float b() const;

	__host__ __device__  const Vec3& operator+() const;
	__host__ __device__  Vec3 operator-() const;
	__host__ __device__  float operator[](int i) const;
	__host__ __device__  float& operator[](int i);

	__host__ __device__  Vec3& operator+=(const Vec3& v2);
	__host__ __device__  Vec3& operator-=(const Vec3& v2);
	__host__ __device__  Vec3& operator*=(const Vec3& v2);
	__host__ __device__  Vec3& operator/=(const Vec3& v2);
	__host__ __device__  Vec3& operator*=(const float t);
	__host__ __device__  Vec3& operator/=(const float t);


	__host__ __device__ static  float dot(const Vec3& v1, const Vec3& v2);
	__host__ __device__ static  Vec3 cross(const Vec3& v1, const Vec3& v2);
	__host__ __device__ static  Vec3 cross2(const Vec3& lhs, const Vec3& rhs);
	__host__ __device__  float Length() const;
	__host__ __device__  float SquaredLength() const;
	__host__ __device__  void MakeUnitVector();
	__host__ __device__  bool isZero();
	__host__ __device__ static  Vec3 unit_vector(Vec3 v);
	__host__ __device__ static  float Distance(Vec3 a, Vec3 b);
};
__host__ __device__  Vec3 operator+(const Vec3& v1, const Vec3& v2);
__host__ __device__  Vec3 operator-(const Vec3& v1, const Vec3& v2);
__host__ __device__  Vec3 operator*(const Vec3& v1, const Vec3& v2);
__host__ __device__  Vec3 operator*(float t, const Vec3& v);
__host__ __device__  Vec3 operator*(const Vec3& v, float t);
__host__ __device__  Vec3 operator/(const Vec3& v1, const Vec3& v2);
__host__ __device__  Vec3 operator/(Vec3 v, float t);



