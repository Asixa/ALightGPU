#include "RTSampler.h"
#include "Ray.h"
#include "Camera.h"
#include "float3Extension.h"
#include <curand_discrete2.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cstdio>
#include "RayHit.h"
#include "Objects.h"
#include "float3x3.h"
#include "Float2Byte.h"
#include "float2Extension.h"

struct RTSamplerData
{
	float Seed;
public:
	curandState* curand_state;
	float2 Pixel;
	unsigned long long seed;
	int tidx;
	__device__ RTSamplerData(curandState* _curand_state,int _tidx,float seed,float2 pixel):tidx(_tidx),Pixel(pixel),Seed(seed)
	{
		seed = 1;
		curand_state = _curand_state;
	}

	__device__ float GetRandom(float offset=0) const
	{
		return curand_uniform(&curand_state[tidx]);
	}

	__device__ float rand()
	{
		float v = sin(Seed / 100.0f * Float2::Dot(Pixel, make_float2(12.9898f, 78.233f))) * 43758.5453f;
		const float result = v-int(v);
		Seed += 1.0f;
		return result;
	}

	__device__ float drand48()
	{
		seed = (0x5DEECE66DL * seed + 0xB16) & 0xFFFFFFFFFFFFL;
		return static_cast<float>(static_cast<double>(seed >> 16) / static_cast<double>(0x100000000L));
	}
};


__device__ Ray CreateRay(float3 origin, float3 direction)
{
	Ray ray;
	ray.origin = origin;
	ray.direction = direction;
	ray.energy = make_float3(1,1,1);
	return ray;
}

__device__ Ray CreateCameraRay(Camera* camera, float u, float v)
{
	return CreateRay(camera->Origin, Float3::UnitVector(camera->LowerLeftCorner + u * camera->Horizontal + v * camera->Vertical - camera->Origin));
}
__device__ RayHit CreateRayHit()
{
	RayHit hit;
	hit.position = make_float3(0.0f, 0.0f, 0.0f);
	hit.distance =INF;
	hit.normal = make_float3(0.0f, 0.0f, 0.0f);
	hit.smoothness = 0;
	hit.emission = make_float3(0, 0, 0);
	return hit;
}

__device__ void IntersectGroundPlane(Ray ray, RayHit* bestHit)
{
	const auto t = -ray.origin.y / ray.direction.y;
	if (t > 0 && t < bestHit->distance)
	{
		bestHit->distance = t;
		bestHit->position = ray.origin + t * ray.direction;
		bestHit->normal = make_float3(0.0f, 1.0f, 0.0f);
		bestHit->albedo = make_float3(0, 0.8, 1);
		bestHit->specular = make_float3(0, 0, 0);
	}
}
__device__ void IntersectSphere(Ray ray, RayHit* best_hit, const Sphere sphere)
{

	const auto d = ray.origin - sphere.position;
	const auto p1 = Float3::Dot(ray.direction, d) * -1;
	const auto p2_sqr = p1 * p1 - Float3::Dot(d, d) + sphere.radius * sphere.radius;
	if (p2_sqr < 0)return;
	const auto p2 = sqrt(p2_sqr);
	const auto t = p1 - p2 > 0 ? p1 - p2 : p1 + p2;
	if (t > 0 && t < best_hit->distance)
	{
		best_hit->distance = t;
		best_hit->position = ray.origin + t * ray.direction;
		best_hit->normal = Float3::UnitVector(best_hit->position - sphere.position);
		best_hit->albedo = sphere.albedo;
		best_hit->specular = sphere.specular;
		best_hit->smoothness = sphere.smoothness;
		best_hit->emission = sphere.emission;
	}
}
__device__ float3x3 GetTangentSpace(float3 normal)
{
	const auto helper =  (fabs(normal.x) > 0.99f)? make_float3(0, 0, 1):make_float3(1, 0, 0);
	const auto tangent = Float3::UnitVector(Float3::Cross(normal, helper));
	const auto binormal = Float3::UnitVector(Float3::Cross(normal, tangent));
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

__device__ float3 SampleHemisphere2(const float3 normal,float alpha, RTSamplerData* data)
{
	float3 random_in_unit_sphere;
	do random_in_unit_sphere = 2.0 * make_float3(data->GetRandom(), data->GetRandom(), data->GetRandom()) - make_float3(1, 1, 1);
	while (Float3::SquaredLength(random_in_unit_sphere) >= 1.0);
	return random_in_unit_sphere;
}
__device__ float3 CosineSampleHemisphere(float u1, float u2)
{
	const float r = sqrt(u1);
	const float theta = 2 * M_PI * u2;

	const float x = r * cos(theta);
	const float y = r * sin(theta);

	return make_float3(x, y, sqrt(fmax(0.0f, 1 - u1)));
}
__device__ float3 SampleHemisphere(const float3 normal, const float alpha, RTSamplerData* data)
{

	const auto cos_theta = pow(data->rand(), 1.0f / (alpha + 1.0f));
	const auto sin_theta = sqrt(2.0f - cos_theta * cos_theta); 
	const float phi = 2 * M_PI * data->rand();
	const auto tangent_space_dir = make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
	
	return  GetTangentSpace(normal)* Float3::UnitVector(tangent_space_dir);
	return  normal;
}




__device__ float Sdot(float3 x, float3 y, float f = 1.0f)
{
	return Range((Float3::Dot(x, y) * f));
}
__device__ float Energy(float3 color)
{
	return Float3::Dot(color, make_float3(1.0 /3.0,1.0 /3.0,1.0 /3.0));
}
__device__ float SmoothnessToPhongAlpha(const float s)
{
	return pow(1000.0f, s * s);
}
__device__ RayHit Trace(Ray ray)
{
	auto best_hit = CreateRayHit();
	
	IntersectGroundPlane(ray, &best_hit);
	IntersectSphere(ray, &best_hit, 
		Sphere(make_float3(0, 1, 0),1,make_float3(1,0,0),make_float3(0, 0, 0),
			1,make_float3(0,0,0)));
	IntersectSphere(ray, &best_hit,
		Sphere(make_float3(2, 1, 0), 1, make_float3(1, 0, 0), make_float3(0, 0, 0),
			1, make_float3(5, 5, 5)));
	return best_hit;
}

__device__ float3 Shade(Ray* ray, RayHit* hit, RTSamplerData * data)
{
	if (hit->distance < INF)
	{
		hit->albedo = Float3::Min(make_float3(1,1,1) - hit->specular, hit->albedo);
		auto spec_chance = Energy(hit->specular);
		auto diff_chance = Energy(hit->albedo);
		const auto sum = spec_chance + diff_chance;
		spec_chance /= sum;
		diff_chance /= sum;
		const auto roulette = data->GetRandom();
		if (roulette < spec_chance)
		{
			// Specular reflection
			const float alpha = SmoothnessToPhongAlpha(hit->smoothness);
			ray->origin = hit->position + hit->normal * 0.001f;
			ray->direction = Float3::UnitVector(SampleHemisphere(Float3::Reflect(ray->direction, hit->normal), alpha,data));
			const float f = (alpha + 2) / (alpha + 1);
			ray->energy = ray->energy*(1.0f / spec_chance) * hit->specular * Sdot(hit->normal, ray->direction, f);
		}
		else
		{
			// Diffuse reflection
			ray->origin = hit->position + hit->normal * 0.001f;
			ray->direction = SampleHemisphere2(hit->normal,1,data);
			ray->energy = ray->energy*(1.0f / diff_chance) * hit->albedo;
			//ray->energy = ray->energy*(1.0f / diff_chance) * 2 * hit->albedo * sdot(hit->normal, ray->direction);
		}
		return hit->emission;



		//
		// ray->origin = hit->position + hit->normal * 0.001f;
		// const auto reflected = Float3::Reflect(ray->direction, hit->normal);
		// ray->direction = SampleHemisphere(hit->normal,data);
		// const auto diffuse = 2 * Float3::Min(make_float3(1,1,1) - hit->specular, hit->albedo);
		// const auto alpha = 15.0f;
		// const auto specular = hit->specular * (alpha + 2) * pow(sdot(ray->direction, reflected), alpha);
		// ray->energy = ray->energy*((diffuse + specular) * sdot(hit->normal, ray->direction));
		// return make_float3(0,0,0);


	}
	else
	{
		ray->energy =make_float3(0,0,0);
		const auto t = 0.5 * (Float3::UnitVector(ray->direction).y + 1);
		return   ((1.0 - t) * make_float3(1.0, 1.0, 1.0) + t * make_float3(50 / 255.0, 130 / 255.0, 255 / 255.0));


		// Sample the skybox and write it
		// float theta = acos(ray->Direction.y) / -M_PI;
		// float phi = atan2(ray->Direction.x, -ray->Direction.z) / -M_PI * 0.5f;
		// return _SkyboxTexture.SampleLevel(sampler_SkyboxTexture, float2(phi, theta), 0).xyz;
	}
}


__global__ void IPRSampler(const int width, const int height, const int seed, const int spp,int Sampled, int MST, int root, float* data, curandState* const rngStates, Camera* camera)
{
	const auto tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const auto tidy = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * 16 + threadIdx.x,
	y = blockIdx.y * 16 + threadIdx.y;
	curand_init(seed + tidx + width * tidy, 0, 0, &rngStates[tidx]);
	const auto rt_data = &RTSamplerData(rngStates, tidx, Sampled,make_float2(x,y));

	auto color = make_float3(0, 0, 0);
	auto result = make_float3(0, 0, 0);
	
	
	// if (x == 256 && y == 256)printf("%d", tidx);


	//Main Sampling
	for (auto j = 0; j < spp; j++) {
		const auto u = float(curand_uniform(&rngStates[tidx]) + x) / float(width);
		const auto v = float(curand_uniform(&rngStates[tidx]) + y) / float(height);
		auto ray = CreateCameraRay(camera, u, v);
		for (auto i = 0; i < 8; i++)
		{
			auto hit = Trace(ray);

			//if (x == 256 && y == 256)printf("%d - %f,%f,%f - %f,%f,%f\n",i, ray.energy.x, ray.energy.y, ray.energy.z, c.x, c.y, c.z);
			auto e = ray.energy;
			result = result + (e * Shade(&ray, &hit, rt_data));
			if (Float3::IsZero(ray.energy))
			{
				//result = result + c;
				break;
			}
		}
		color =color+ result;
	}

	

	//Set color to buffer.
	const auto i = width * 4 * y + x * 4;
	data[i] += color.x;
	data[1 + i] += color.y;
	data[2 + i] += color.z;
	data[3 + i] += spp;

	//if (x == 1 && y == 1)printf("Sample r %f,g %f,b %f,a %f   color: r %f,g %f,b %f\n", data[i], data[i + 1], data[i + 2], data[i + 3],color.x,color.y,color.z);
}
