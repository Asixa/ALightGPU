#pragma once
#include "float3Extension.h"
#include <curand_kernel.h>
#include "float2Extension.h"

struct RTSamplerData
{
	float3 ambient;
	int max_depth;

	float Seed;
public:
	curandState* curand_state;
	float2 Pixel;
	unsigned long long seed;
	int tidx;
	__device__ RTSamplerData(curandState* _curand_state, int _tidx, float seed, float2 pixel) :tidx(_tidx), Pixel(pixel), Seed(seed)
	{
		seed = 1;
		curand_state = _curand_state;
	}

	__device__ float GetRandom(float offset = 0) const
	{
		return curand_uniform(&curand_state[tidx]);
	}

	__device__ float rand()
	{
		float v = sin(Seed / 100.0f * Float2::Dot(Pixel, make_float2(12.9898f, 78.233f))) * 43758.5453f;
		const float result = v - int(v);
		Seed += 1.0f;
		return result;
	}

	__device__ float drand48()
	{
		seed = (0x5DEECE66DL * seed + 0xB16) & 0xFFFFFFFFFFFFL;
		return static_cast<float>(static_cast<double>(seed >> 16) / static_cast<double>(0x100000000L));
	}
};
