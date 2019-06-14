#include "RTSampler.h"
#include "Ray.h"
#include "Camera.h"
#include <curand_discrete2.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cstdio>
#include "RayHit.h"
#include "Objects.h"
#include "float3x3.h"
#include "Float2Byte.h"
#include "PathRenderer.h"
#include "SimpleRenderer.h"


__device__ Ray CreateCameraRay(Camera* camera, float u, float v)
{
	return Ray(camera->Origin, normalize(camera->LowerLeftCorner + u * camera->Horizontal + v * camera->Vertical - camera->Origin));
}

// __device__ RayHit CreateRayHit()
// {
// 	RayHit hit;
// 	hit.position = make_float3(0.0f, 0.0f, 0.0f);
// 	hit.distance =INF;
// 	hit.normal = make_float3(0.0f, 0.0f, 0.0f);
// 	hit.smoothness = 0;
// 	hit.emission = make_float3(0, 0, 0);
// 	return hit;
// }

__device__ void IntersectGroundPlane(Ray ray, SurfaceHitRecord* bestHit, const RTDeviceData* data)
{
	const auto t = -ray.origin.y / ray.direction.y;
	if (t > 0 && t < bestHit->t)
	{
		bestHit->t = t;
		bestHit->p = ray.origin + t * ray.direction;
		bestHit->normal = make_float3(0.0f, 1.0f, 0.0f);
		bestHit->mat_ptr = &data->Materials[0];
		// bestHit->albedo = make_float3(0, 0.8, 1);
		// bestHit->specular = make_float3(0, 0, 0);
	}
}
__device__ void IntersectSphere(Ray ray, SurfaceHitRecord* best_hit, const Sphere sphere,const RTDeviceData* data)
{
	const auto d = ray.origin - sphere.position;
	const auto p1 = dot(ray.direction, d) * -1;
	const auto p2_sqr = p1 * p1 - dot(d, d) + sphere.radius * sphere.radius;
	if (p2_sqr < 0)return;
	const auto p2 = sqrt(p2_sqr);
	const auto t = p1 - p2 > 0 ? p1 - p2 : p1 + p2;
	if (t > 0 && t < best_hit->t)
	{
		best_hit->t = t;
		best_hit->p = ray.origin + t * ray.direction;
		best_hit->normal = normalize(best_hit->p - sphere.position);
		best_hit->mat_ptr =&data->Materials[1];
		// best_hit->albedo = sphere.albedo;
		// best_hit->specular = sphere.specular;
		// best_hit->smoothness = sphere.smoothness;
		// best_hit->emission = sphere.emission;
	}
}
__device__ SurfaceHitRecord Trace(const Ray ray,const RTDeviceData* data)
{
	auto best_hit = SurfaceHitRecord();
	IntersectGroundPlane(ray, &best_hit,data);
	//IntersectSphere(ray, &best_hit,Sphere(make_float3(0, 1, 0), 1, make_float3(1, 0, 0), make_float3(0, 0, 0), 1, make_float3(0, 0, 0)),data);
	//IntersectSphere(ray, &best_hit,Sphere(make_float3(2, 1, 0), 1, make_float3(1, 0, 0), make_float3(0, 0, 0), 1, make_float3(5, 5, 5)), data);
	return best_hit;
}

__device__ float3x3 GetTangentSpace(float3 normal)
{
	const auto helper =  (fabs(normal.x) > 0.99f)? make_float3(0, 0, 1):make_float3(1, 0, 0);
	const auto tangent = normalize(cross(normal, helper));
	const auto binormal = normalize(cross(normal, tangent));
	return float3x3(tangent, binormal, normal);
}

// __device__ float3 SampleHemisphere(const float3 normal,float alpha,RTSamplerData* data)
// {
// 	const auto cos_theta =data->rand();
// 	const auto sin_theta = sqrt(fmax(0.0f, 1.0f - cos_theta * cos_theta));
// 	const float phi = 2 * M_PI * data->rand();
// 	const auto tangent_space_dir = make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
// 	// if (curand_uniform(data->curand_state) < 0.1f)printf("%f,%f,%f,   %f,%f,%f\n", tangent_space_dir.x, tangent_space_dir.y, tangent_space_dir.z,
// 	// 	normal.x, normal.y, normal.z);
// 	
// 	return GetTangentSpace(normal)* tangent_space_dir;
//
// }





__device__ float Sdot(float3 x, float3 y, float f = 1.0f)
{
	return Range((dot(x, y) * f));
}
__device__ float Energy(float3 color)
{
	return dot(color, make_float3(1.0 /3.0,1.0 /3.0,1.0 /3.0));
}
__device__ float SmoothnessToPhongAlpha(const float s)
{
	return pow(1000.0f, s * s);
}


__global__ void IPRSampler(const int width, const int height, const int seed, const int spp,int Sampled, int MST, int root, float* output, curandState* const rngStates, Camera* camera,RTHostData host_data)
{
	const auto tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const auto tidy = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
	curand_init(seed + tidx + width * tidy, 0, 0, &rngStates[tidx]);
	const auto data = &RTDeviceData(rngStates, tidx, Sampled,make_float2(x,y));
	data->Materials = host_data.Materials;
	auto color = make_float3(0, 0, 0);
	auto result = make_float3(0, 0, 0);




	for (auto j = 0; j < spp; j++)
	{
		const auto u =( curand_uniform(&rngStates[tidx]) + x) / width;
		const auto v = (curand_uniform(&rngStates[tidx]) + y) / height;
		auto ray = CreateCameraRay(camera, u, v);

		//printf("%f,%f\n", u, v);
		// printf("%f,%f,%f\n",camera->Horizontal.x, camera->Horizontal.y, camera->Horizontal.z);
		// printf("%f,%f,%f\n",camera->Origin.x, camera->Origin.y, camera->Origin.z);
		// printf("%f,%f,%f\n",camera->Vertical.x, camera->Vertical.y, camera->Vertical.z);
		//printf("%f,%f,%f\n",camera->LowerLeftCorner.x, camera->LowerLeftCorner.y, camera->LowerLeftCorner.z);
		float3 factor = make_float3(1, 1, 1);
		for (auto i = 0; i < 1; i++)
		{
			auto hit = Trace(ray,data);
			result = RTRenderer::SimpleRenderer::Shade(ray, hit, 0, 9999, 1, i, factor, data);
		}


		color += result;
		//printf("%f,%f,%f\n", result.x, result.y, result.z);
		//color = make_float3(228/256.0,0,127/256.0);
		result = make_float3(0, 0, 0);
	}

	

	//Set color to buffer.
	const auto i = width * 4 * y + x * 4;
	output[i] += color.x;
	output[1 + i] += color.y;
	output[2 + i] += color.z;
	output[3 + i] += spp;
}
