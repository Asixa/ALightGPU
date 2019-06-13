#pragma once
#include "float2Extension.h"
struct RTSamplerData;
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
			float3& factor, const RTSamplerData* data
		);
	}
}
