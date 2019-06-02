#pragma once
#include "curand_kernel.h"

namespace RayTracer
{
	float* h_pixeldataF;
	float* d_pixeldata;
	curandState* d_rng_states = nullptr;
}