#pragma once
#include "Ray.h"
#include "SurfaceHitRecord.h"


#include "RTDeviceData.h"

namespace RTRenderer
{
	namespace PathRenderer{

		__device__ float3 Shade(
			Ray & r,      // Ray being sent
			SurfaceHitRecord & rec,
			//const Scene* s,
			float tmin,        // minimum hit parameter to be searched for
			float tmax,        // maximum hit parameter to be searched for
			float time,
			float2 & sseed,
			float2 & rseed,
			int depth,
			int spec_depth,
			bool corl,         // count only reflected light? (not emitted)
			//PhotonMaps* maps
			float3 & factor, const RTDeviceData * data
		);
	}
}
