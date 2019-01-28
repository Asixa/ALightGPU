#pragma once
#include <crt/host_defines.h>
#include <iostream>
class Vec2
{
public:
	__device__ __host__ Vec2() {}
	__device__ __host__ Vec2(float e0, float e1) { X = e0; Y = e1; }

	__device__ __host__ inline const Vec2& operator+() const { return *this; }
	__device__ __host__ inline Vec2 operator-() const { return Vec2(-X, -Y); }

	__device__ __host__ inline Vec2& operator+=(const Vec2 &v2);
	__device__ __host__ inline Vec2& operator-=(const Vec2 &v2);
	__device__ __host__ inline Vec2& operator*=(const Vec2 &v2);
	__device__ __host__ inline Vec2& operator/=(const Vec2 &v2);
	__device__ __host__ inline Vec2& operator*=(const float t);
	__device__ __host__ inline Vec2& operator/=(const float t);

	__device__ __host__ inline float length() const { return sqrt(X * X + Y * Y); }
	__device__ __host__ inline float squared_length() const { return X * X + Y * Y; }
	__device__ __host__ inline void Normalize();

	float X, Y;
};

#pragma region Vec2

__device__ inline std::istream& operator>>(std::istream &is, Vec2 &t)
{
	is >> t.X >> t.Y;
	return is;
}

__device__ inline std::ostream& operator<<(std::ostream &os, const Vec2 &t)
{
	os << t.X << " " << t.Y;
	return os;
}

__device__ inline void Vec2::Normalize()
{
	float k = 1.0f / sqrt(X * X + Y * Y);
	X *= k; Y *= k;
}


__device__ inline Vec2 operator+(const Vec2 &v1, const Vec2 &v2)
{
	return {v1.X + v2.X, v1.Y + v2.Y};
}

__device__ inline Vec2 operator-(const Vec2 &v1, const Vec2 &v2)
{
	return {v1.X - v2.X, v1.Y - v2.Y};
}

__device__ inline Vec2 operator*(const Vec2 &v1, const Vec2 &v2)
{
	return {v1.X * v2.X, v1.Y * v2.Y};
}

__device__ inline Vec2 operator/(const Vec2 &v1, const Vec2 &v2)
{
	return {v1.X / v2.X, v1.Y / v2.Y};
}

__device__ inline Vec2 operator*(float t, const Vec2 &v)
{
	return {t*v.X, t*v.Y};
}

__device__ inline Vec2 operator/(Vec2 v, float t)
{
	return {v.X / t, v.Y / t};
}

__device__ inline Vec2 operator*(const Vec2 &v, float t)
{
	return {t*v.X, t*v.Y};
}

__device__ inline float dot(const Vec2 &v1, const Vec2 &v2)
{
	return v1.X * v2.X + v1.Y * v2.Y;
}

__device__ inline Vec2& Vec2::operator+=(const Vec2 &v)
{
	X += v.X;
	Y += v.Y;
	return *this;
}

__device__ inline Vec2& Vec2::operator*=(const Vec2 &v)
{
	X *= v.X;
	Y *= v.Y;
	return *this;
}

__device__ inline Vec2& Vec2::operator/=(const Vec2 &v)
{
	X /= v.X;
	Y /= v.Y;
	return *this;
}

__device__ inline Vec2& Vec2::operator-=(const Vec2& v)
{
	X -= v.X;
	Y -= v.Y;
	return *this;
}

__device__ inline Vec2& Vec2::operator*=(const float t)
{
	X *= t;
	Y *= t;
	return *this;
}

__device__ inline Vec2& Vec2::operator/=(const float t)
{
	const auto k = 1.0f / t;
	X *= k;
	Y *= k;
	return *this;
}

__device__ inline Vec2 Normalize(Vec2 v)
{
	return v / v.length();
}

#pragma endregion