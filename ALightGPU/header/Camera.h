#pragma once
#include "vec3.h"
#include "ray.h"
#include "root.h"
#include <curand_kernel.h>


class Camera
{
public:
	float data[12];
	__host__ __device__ Camera(){}
	__host__ __device__ Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float fov, float aspect)
	{

		Vec3 u, v, w;
		float theta = fov * M_PI / 180;
		float half_height = tan(theta / 2);
		float half_width = aspect*half_height;
		auto Origin = lookfrom;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		auto LowerLeftCorner = Vec3(-half_width, -half_height, - 1.0);
		 LowerLeftCorner = Origin - half_width * u - half_height * v - w;
		auto Horizontal = 2 * half_width*u;
		auto Vertical = 2 * half_height*v;

		data[0] = Origin.x();
		data[1] = Origin.y();
		data[2] = Origin.z();

		data[3] = LowerLeftCorner.x();
		data[4] = LowerLeftCorner.y();
		data[5] = LowerLeftCorner.z();

		data[6] = Horizontal.x();
		data[7] = Horizontal.y();
		data[8] = Horizontal.z();

		data[9] = Vertical.x();
		data[10] = Vertical.y();
		data[11] = Vertical.z();

		// data[12] = u.x();
		// data[13] = u.y();
		// data[14] = u.z();
		//
		// data[15] = v.x();
		// data[16] = v.y();
		// data[17] = v.z();
		//
		// data[18] = w.x();
		// data[19] = w.y();
		// data[20] = w.z();

		// printf("origin  %f,%f,%f LowerLeftCorner  %f,%f,%f  Horizontal()  %f,%f,%f Vertical %f,%f,%f\n",
		// 	Origin.x(), Origin.y(), Origin.z(), 
		// 	LowerLeftCorner.x(), LowerLeftCorner.y(), LowerLeftCorner.z(),
		// 	Horizontal.x(), Horizontal.y(), Horizontal.z(),
		// 	Vertical.x(), Vertical.y(), Vertical.z());
		//
		// printf("origin  %f,%f,%f LowerLeftCorner  %f,%f,%f  Horizontal()  %f,%f,%f Vertical %f,%f,%f\n",
		// 	data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11]);

	}
	__device__ Vec3 Origin() { return  Vec3(data[0], data[1], data[2]); }

	__device__ Vec3 LowerLeftCorner() { return Vec3(data[3], data[4], data[5]); }

	__device__ Vec3 Horizontal() { return Vec3(data[6], data[7], data[8]); }


	__device__ Vec3 Vertical() { return Vec3(data[9], data[10], data[11]); }

	//
	// __device__ Vec3 u() { return Vec3(data[12], data[13], data[14]); }
	//
	//
	// __device__ Vec3 v() { return Vec3(data[15], data[16], data[17]); }
	//
	//
	// __device__ Vec3 w() { return Vec3(data[18], data[19], data[20]); }


	__device__ Ray GetRay(float u,float v) const
	{
		//return Ray(Origin, LowerLeftCorner + u * Horizontal + v * Vertical);
		return Ray(Vec3(0, 0, 0), Vec3(-2.0, -2.0, -1.0) + u * Vec3(4, 0, 0) + v * Vec3(0, 4, 0));
	}



};
