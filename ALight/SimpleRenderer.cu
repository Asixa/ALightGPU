#include "SimpleRenderer.h"
#include "SurfaceHitRecord.h"
#include "RTDeviceData.h"
#include "Material.h"
#include "Ray.h"
#include "MathHelper.h"
#include <cstdio>

float3 RTRenderer::SimpleRenderer::Shade(Ray& r, SurfaceHitRecord& rec, float tmin, float tmax, float time,int depth,
	float3& factor, const RTDeviceData* data)
{
	float3 c = make_float3(0, 0, 0);
	if (rec.t < 99998)
	{
		//printf("%f", rec.t);
		return make_float3(1, 0, 0);
		float3 random_in_unit_sphere;
		do random_in_unit_sphere = 2.0 * make_float3(data->GetRandom(), data->GetRandom(), data->GetRandom())- make_float3(1, 1, 1);
		while (squaredLength(random_in_unit_sphere) >= 1.0);
		auto scattered=Ray();
		float3 attenuation;
		if (depth < 8 && rec.mat_ptr->scatter(r, rec, attenuation, scattered, random_in_unit_sphere,data))
		{
			factor *= attenuation;
			r = scattered;
		}
	}
	else
	{

		const auto t = 0.5f * (normalize(r.direction).y + 1);
		
		c = ((1.0 - t) * make_float3(1.0, 1.0, 1.0) + t * make_float3(50 / 255.0, 130 / 255.0, 255 / 255.0));
		return  c;
	}
	//return c;
}
