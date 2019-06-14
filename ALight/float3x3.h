#pragma once
#include "float3Extension.h"
#include <cmath>

struct float3x3
{
	union
	{
		struct
		{
			float D00;
			float D01;
			float D02;
			float D10;
			float D11;
			float D12;
			float D20;
			float D21;
			float D22;
		};
		float data[3][3];
	};

public:
	 __host__ __device__ float3x3();
	 __host__ __device__ float3x3(float data[]);
	 __host__ __device__  float3x3(float3 row0, float3 row1, float3 row2);
	 __host__ __device__  float3x3(float d00, float d01, float d02, float d10, float d11,
		float d12, float d20, float d21, float d22);

	 __host__ __device__ static  float3x3 Identity();
	 __host__ __device__ static  float3x3 zero();
	 __host__ __device__ static  float3x3 One();

	/**
	* Returns the determinate of a matrix.
	* @param matrix: The input matrix.
	* @return: A scalar value.
	*/
	 __host__ __device__ static  float Determinate(float3x3 matrix);

	/**
	 * Converts a quaternion to a rotation matrix.
	 * @param rotation: The input quaternion.
	 * @return: A new rotation matrix.
	 */
	
	//static  Matrix3x3 FromQuaternion(Quaternion rotation);

	/**
	 * Returns the inverse of a matrix.
	 * @param matrix: The input matrix.
	 * @return: A new matrix.
	 */
	 __host__ __device__ static  float3x3 Inverse(float3x3 matrix);

	/**
	 * Returns true if a matrix is invertible.
	 * @param matrix: The input matrix.
	 * @return: A new matrix.
	 */
	 __host__ __device__ static  bool IsInvertible(float3x3 matrix);

	/**
	 * Multiplies two matrices element-wise.
	 * @param a: The left-hand side of the multiplication.
	 * @param b: The right-hand side of the multiplication.
	 * @return: A new matrix.
	 */
	 __host__ __device__ static  float3x3 Scale(float3x3 a, float3x3 b);

	/**
	 * Converts a rotation matrix to a quaternion.
	 * @param rotation: The input rotation matrix.
	 * @return: A new quaternion.
	 */

	//static  Quaternion ToQuaternion(Matrix3x3 rotation);

	/**
	 * Returns the transpose of a matrix.
	 * @param matrix: The input matrix.
	 * @return: A new matrix.
	 */
	 __host__ __device__ static  float3x3 Transpose(float3x3 matrix);

	/**
	 * Operator overloading.
	 */
	 __host__ __device__ float3x3& operator+=(const float rhs);
	 __host__ __device__  float3x3& operator-=(const float rhs);
	 __host__ __device__  float3x3& operator*=(const float rhs);
	 __host__ __device__  float3x3& operator/=(const float rhs);
	 __host__ __device__  float3x3& operator+=(const float3x3 rhs);
	 __host__ __device__  float3x3& operator-=(const float3x3 rhs);
	 __host__ __device__  float3x3& operator*=(const float3x3 rhs);
};

__host__ __device__ float3x3 operator-(float3x3 rhs);
__host__ __device__  float3x3 operator+(float3x3 lhs, const float rhs);
__host__ __device__ float3x3 operator-(float3x3 lhs, const float rhs);
__host__ __device__ float3x3 operator*(float3x3 lhs, const float rhs);
__host__ __device__ float3x3 operator/(float3x3 lhs, const float rhs);
__host__ __device__ float3x3 operator+(const float lhs, float3x3 rhs);
__host__ __device__ float3x3 operator-(const float lhs, float3x3 rhs);
__host__ __device__  float3x3 operator*(const float lhs, float3x3 rhs);
__host__ __device__ float3x3 operator+(float3x3 lhs, const float3x3 rhs);
__host__ __device__  float3x3 operator-(float3x3 lhs, const float3x3 rhs);
__host__ __device__  float3x3 operator*(float3x3 lhs, const float3x3 rhs);
__host__ __device__ float3 operator*(float3x3 lhs, const float3 rhs);
__host__ __device__ bool operator==(const float3x3 lhs, const float3x3 rhs);
__host__ __device__ bool operator!=(const float3x3 lhs, const float3x3 rhs);


//***********************************************************************
float3x3::float3x3() : D00(1), D01(0), D02(0), D10(0), D11(1), D12(0), D20(0),
D21(0), D22(1) {}
float3x3::float3x3(float data[]) : D00(data[0]), D01(data[1]), D02(data[2]),
D10(data[3]), D11(data[4]), D12(data[5]), D20(data[6]), D21(data[7]),
D22(data[8]) {}
float3x3::float3x3(float3 row0, float3 row1, float3 row2) : D00(row0.x),
D01(row0.y), D02(row0.z), D10(row1.x), D11(row1.y), D12(row1.z),
D20(row2.x), D21(row2.y), D22(row2.z) {}
float3x3::float3x3(float d00, float d01, float d02, float d10, float d11,
	float d12, float d20, float d21, float d22) : D00(d00), D01(d01),
	D02(d02), D10(d10), D11(d11), D12(d12), D20(d20), D21(d21), D22(d22) {}


float3x3 float3x3::Identity()
{
	return float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
}

float3x3 float3x3::zero()
{
	return float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
}

float3x3 float3x3::One()
{
	return float3x3(1, 1, 1, 1, 1, 1, 1, 1, 1);
}


float float3x3::Determinate(float3x3 matrix)
{
	float v1 = matrix.D00 * (matrix.D22 * matrix.D11 -
		matrix.D21 * matrix.D12);
	float v2 = matrix.D10 * (matrix.D22 * matrix.D01 -
		matrix.D21 * matrix.D02);
	float v3 = matrix.D20 * (matrix.D12 * matrix.D01 -
		matrix.D11 * matrix.D02);
	return v1 - v2 + v3;
}

// Matrix3x3 Matrix3x3::FromQuaternion(Quaternion rotation)
// {
// 	Matrix3x3 m;
// 	float sqw = rotation.W * rotation.W;
// 	float sqx = rotation.x * rotation.x;
// 	float sqy = rotation.y * rotation.y;
// 	float sqz = rotation.z * rotation.z;
//
// 	float invSqr = 1 / (sqx + sqy + sqz + sqw);
// 	m.D00 = (sqx - sqy - sqz + sqw) * invSqr;
// 	m.D11 = (-sqx + sqy - sqz + sqw) * invSqr;
// 	m.D22 = (-sqx - sqy + sqz + sqw) * invSqr;
//
// 	float tmp1 = rotation.x * rotation.y;
// 	float tmp2 = rotation.z * rotation.W;
// 	m.D10 = 2.0 * (tmp1 + tmp2) * invSqr;
// 	m.D01 = 2.0 * (tmp1 - tmp2) * invSqr;
//
// 	tmp1 = rotation.x * rotation.z;
// 	tmp2 = rotation.y * rotation.W;
// 	m.D20 = 2.0 * (tmp1 - tmp2) * invSqr;
// 	m.D02 = 2.0 * (tmp1 + tmp2) * invSqr;
// 	tmp1 = rotation.y * rotation.z;
// 	tmp2 = rotation.x * rotation.W;
// 	m.D21 = 2.0 * (tmp1 + tmp2) * invSqr;
// 	m.D12 = 2.0 * (tmp1 - tmp2) * invSqr;
// 	return m;
// }

float3x3 float3x3::Inverse(float3x3 matrix)
{
	float3x3 a;
	a.D00 = matrix.D22 * matrix.D11 - matrix.D21 * matrix.D12;
	a.D01 = matrix.D21 * matrix.D02 - matrix.D22 * matrix.D01;
	a.D02 = matrix.D12 * matrix.D01 - matrix.D11 * matrix.D02;
	a.D10 = matrix.D20 * matrix.D12 - matrix.D22 * matrix.D10;
	a.D11 = matrix.D22 * matrix.D00 - matrix.D20 * matrix.D02;
	a.D12 = matrix.D10 * matrix.D02 - matrix.D12 * matrix.D00;
	a.D20 = matrix.D21 * matrix.D10 - matrix.D20 * matrix.D11;
	a.D21 = matrix.D20 * matrix.D01 - matrix.D21 * matrix.D00;
	a.D22 = matrix.D11 * matrix.D00 - matrix.D10 * matrix.D01;
	return 1 / Determinate(matrix) * a;
}

bool float3x3::IsInvertible(float3x3 matrix)
{
	return fabs(Determinate(matrix)) > 0.00001;
}

float3x3 float3x3::Scale(float3x3 a, float3x3 b)
{
	float3x3 m;
	m.D00 = a.D00 * b.D00;
	m.D01 = a.D01 * b.D01;
	m.D02 = a.D02 * b.D02;
	m.D10 = a.D10 * b.D10;
	m.D11 = a.D11 * b.D11;
	m.D12 = a.D12 * b.D12;
	m.D20 = a.D20 * b.D20;
	m.D21 = a.D21 * b.D21;
	m.D22 = a.D22 * b.D22;
	return m;
}

// Quaternion Matrix3x3::ToQuaternion(Matrix3x3 rotation)
// {
// 	Quaternion q;
// 	float trace = rotation.D00 + rotation.D11 + rotation.D22;
// 	if (trace > 0)
// 	{
// 		float s = 0.5 / sqrt(trace + 1);
// 		q.W = 0.25 / s;
// 		q.x = (rotation.D21 - rotation.D12) * s;
// 		q.y = (rotation.D02 - rotation.D20) * s;
// 		q.z = (rotation.D10 - rotation.D01) * s;
// 	}
// 	else
// 	{
// 		if (rotation.D00 > rotation.D11 && rotation.D00 > rotation.D22)
// 		{
// 			float s = 2 * sqrt(1 + rotation.D00 - rotation.D11 - rotation.D22);
// 			q.W = (rotation.D21 - rotation.D12) / s;
// 			q.x = 0.25 * s;
// 			q.y = (rotation.D01 + rotation.D10) / s;
// 			q.z = (rotation.D02 + rotation.D20) / s;
// 		}
// 		else if (rotation.D11 > rotation.D22)
// 		{
// 			float s = 2 * sqrt(1 + rotation.D11 - rotation.D00 - rotation.D22);
// 			q.W = (rotation.D02 - rotation.D20) / s;
// 			q.x = (rotation.D01 + rotation.D10) / s;
// 			q.y = 0.25 * s;
// 			q.z = (rotation.D12 + rotation.D21) / s;
// 		}
// 		else
// 		{
// 			float s = 2 * sqrt(1 + rotation.D22 - rotation.D00 - rotation.D11);
// 			q.W = (rotation.D10 - rotation.D01) / s;
// 			q.x = (rotation.D02 + rotation.D20) / s;
// 			q.y = (rotation.D12 + rotation.D21) / s;
// 			q.z = 0.25 * s;
// 		}
// 	}
// 	return q;
// }

float3x3 float3x3::Transpose(float3x3 matrix)
{
	float tmp;
	tmp = matrix.D01;
	matrix.D01 = matrix.D10;
	matrix.D10 = tmp;
	tmp = matrix.D02;
	matrix.D02 = matrix.D20;
	matrix.D20 = tmp;
	tmp = matrix.D12;
	matrix.D12 = matrix.D21;
	matrix.D21 = tmp;
	return matrix;
}


struct float3x3& float3x3::operator+=(const float rhs)
{
	D00 += rhs; D01 += rhs; D02 += rhs;
	D10 += rhs; D11 += rhs; D12 += rhs;
	D20 += rhs; D21 += rhs; D22 += rhs;
	return *this;
}

struct float3x3& float3x3::operator-=(const float rhs)
{
	D00 -= rhs; D01 -= rhs; D02 -= rhs;
	D10 -= rhs; D11 -= rhs; D12 -= rhs;
	D20 -= rhs; D21 -= rhs; D22 -= rhs;
	return *this;
}

struct float3x3& float3x3::operator*=(const float rhs)
{
	D00 *= rhs; D01 *= rhs; D02 *= rhs;
	D10 *= rhs; D11 *= rhs; D12 *= rhs;
	D20 *= rhs; D21 *= rhs; D22 *= rhs;
	return *this;
}

struct float3x3& float3x3::operator/=(const float rhs)
{
	D00 /= rhs; D01 /= rhs; D02 /= rhs;
	D10 /= rhs; D11 /= rhs; D12 /= rhs;
	D20 /= rhs; D21 /= rhs; D22 /= rhs;
	return *this;
}

struct float3x3& float3x3::operator+=(const float3x3 rhs)
{
	D00 += rhs.D00; D01 += rhs.D01; D02 += rhs.D02;
	D10 += rhs.D10; D11 += rhs.D11; D12 += rhs.D12;
	D20 += rhs.D20; D21 += rhs.D21; D22 += rhs.D22;
	return *this;
}

struct float3x3& float3x3::operator-=(const float3x3 rhs)
{
	D00 -= rhs.D00; D01 -= rhs.D01; D02 -= rhs.D02;
	D10 -= rhs.D10; D11 -= rhs.D11; D12 -= rhs.D12;
	D20 -= rhs.D20; D21 -= rhs.D21; D22 -= rhs.D22;
	return *this;
}

struct float3x3& float3x3::operator*=(const float3x3 rhs)
{
	float3x3 m;
	m.D00 = D00 * rhs.D00 + D01 * rhs.D10 + D02 * rhs.D20;
	m.D01 = D00 * rhs.D01 + D01 * rhs.D11 + D02 * rhs.D21;
	m.D02 = D00 * rhs.D02 + D01 * rhs.D12 + D02 * rhs.D22;
	m.D10 = D10 * rhs.D00 + D11 * rhs.D10 + D12 * rhs.D20;
	m.D11 = D10 * rhs.D01 + D11 * rhs.D11 + D12 * rhs.D21;
	m.D12 = D10 * rhs.D02 + D11 * rhs.D12 + D12 * rhs.D22;
	m.D20 = D20 * rhs.D00 + D21 * rhs.D10 + D22 * rhs.D20;
	m.D21 = D20 * rhs.D01 + D21 * rhs.D11 + D22 * rhs.D21;
	m.D22 = D20 * rhs.D02 + D21 * rhs.D12 + D22 * rhs.D22;
	*this = m;
	return *this;
}

float3x3 operator-(float3x3 rhs) { return rhs * -1; }
float3x3 operator+(float3x3 lhs, const float rhs) { return lhs += rhs; }
float3x3 operator-(float3x3 lhs, const float rhs) { return lhs -= rhs; }
float3x3 operator*(float3x3 lhs, const float rhs) { return lhs *= rhs; }
float3x3 operator/(float3x3 lhs, const float rhs) { return lhs /= rhs; }
float3x3 operator+(const float lhs, float3x3 rhs) { return rhs += lhs; }
float3x3 operator-(const float lhs, float3x3 rhs) { return rhs -= lhs; }
float3x3 operator*(const float lhs, float3x3 rhs) { return rhs *= lhs; }
float3x3 operator+(float3x3 lhs, const float3x3 rhs) { return lhs += rhs; }
float3x3 operator-(float3x3 lhs, const float3x3 rhs) { return lhs -= rhs; }
float3x3 operator*(float3x3 lhs, const float3x3 rhs) { return lhs *= rhs; }

float3 operator*(float3x3 lhs, const float3 rhs)
{
	float3 v;
	v.x = lhs.D00 * rhs.x + lhs.D01 * rhs.y + lhs.D02 * rhs.z;
	v.y = lhs.D10 * rhs.x + lhs.D11 * rhs.y + lhs.D12 * rhs.z;
	v.z = lhs.D20 * rhs.x + lhs.D21 * rhs.y + lhs.D22 * rhs.z;
	return v;
}

bool operator==(const float3x3 lhs, const float3x3 rhs)
{
	return lhs.D00 == rhs.D00 &&
		lhs.D01 == rhs.D01 &&
		lhs.D02 == rhs.D02 &&
		lhs.D10 == rhs.D10 &&
		lhs.D11 == rhs.D11 &&
		lhs.D12 == rhs.D12 &&
		lhs.D20 == rhs.D20 &&
		lhs.D21 == rhs.D21 &&
		lhs.D22 == rhs.D22;
}

bool operator!=(const float3x3 lhs, const float3x3 rhs)
{
	return !(lhs == rhs);
}