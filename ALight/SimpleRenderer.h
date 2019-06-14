#pragma once
#include "cuda_runtime.h"
struct RTDeviceData;
class SurfaceHitRecord;
struct Ray;

namespace RTRenderer
{
	namespace SimpleRenderer {

		__device__ float3 Shade(
			Ray& r,      // Ray being sent
			SurfaceHitRecord& rec,
			float tmin,       
			float tmax,        
			float time,
			int depth,
			float3& factor, const RTDeviceData* data
		);
	}
}
