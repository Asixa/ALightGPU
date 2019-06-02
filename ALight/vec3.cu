#include "vec3.h"
// ReSharper disable CppInconsistentNaming
#include <cuda_runtime_api.h>
#include <math.h>

__host__ __device__ Vec3::Vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
__host__ __device__ Vec3::Vec3() {}
__host__ __device__ float Vec3::x() const { return e[0]; }
__host__ __device__ float Vec3::y() const { return e[1]; }
__host__ __device__ float Vec3::z() const { return e[2]; }
__host__ __device__ float Vec3::r() const { return e[0]; }
__host__ __device__ float Vec3::g() const { return e[1]; }
__host__ __device__ float Vec3::b() const { return e[2]; }

const Vec3& Vec3::operator+() const { return *this; }
Vec3 Vec3::operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
float Vec3::operator[](int i) const { return e[i]; }
float& Vec3::operator[](int i) { return e[i]; }


Vec3& Vec3::operator+=(const Vec3 & v) {
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
	
}

Vec3& Vec3::operator*=(const Vec3 & v) {
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	return *this;
}

Vec3& Vec3::operator/=(const Vec3 & v) {
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	return *this;
}

Vec3& Vec3::operator-=(const Vec3 & v) {
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}

Vec3& Vec3::operator*=(const float t) {
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

Vec3& Vec3::operator/=(const float t) {
	float k = 1.0 / t;
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	return *this;
}

float Vec3::dot(const Vec3 & v1, const Vec3 & v2) { return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2]; }
Vec3 Vec3::cross(const Vec3 & v1, const Vec3 & v2)
{
	return Vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
		(-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
		(v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}
Vec3 Vec3::cross2(const Vec3 & lhs, const Vec3 & rhs)
{
	return Vec3(lhs.y() * rhs.z() - lhs.z() * rhs.y(),
		lhs.z() * rhs.x() - lhs.x() * rhs.z(),
		lhs.x() * rhs.y() - lhs.y() * rhs.x());
}
float Vec3::Length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }

float Vec3::SquaredLength() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

bool Vec3::isZero() { return e[0] == 0 && e[1] == 0 && e[2] == 0; }

Vec3 Vec3::unit_vector(Vec3 v) { return v / v.Length(); }

float Vec3::Distance(Vec3 a, Vec3 b) { return (a - b).Length(); }

void Vec3::MakeUnitVector() {
	const float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	e[0] *= k; e[1] *= k; e[2] *= k;
}

Vec3 operator+(const Vec3 & v1, const Vec3 & v2) { return Vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]); }

Vec3 operator-(const Vec3 & v1, const Vec3 & v2) { return Vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]); }

Vec3 operator*(const Vec3 & v1, const Vec3 & v2) { return Vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]); }

Vec3 operator/(const Vec3 & v1, const Vec3 & v2) { return Vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]); }

Vec3 operator*(float t, const Vec3 & v) { return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]); }

Vec3 operator/(Vec3 v, float t) { return Vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t); }

Vec3 operator*(const Vec3 & v, float t) { return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]); }
 