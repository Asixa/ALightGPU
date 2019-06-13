#include "PathRenderer.h"
#include "SurfaceHitRecord.h"
#include "Material.h"

//
// float3 PathRenderer::Shade(const Ray& r, SurfaceHitRecord& rec, float tmin, float tmax, float time, float2& sseed, float2& rseed, int depth, int spec_depth, bool corl,
// 	float& brdf_scale, float3& R, RTSamplerData& data)
// {
// 	bool dummy_bool = true;
// 	// float brdf_scale;		
// 	// float3 R;
// 	if (rec.t < 99999)
// 	{
// 		float3 v_out;
// 		auto c = make_float3(0, 0, 0);
// 		c += rec.mat_ptr->emittedRadiance(rec.uvw, -r.direction, rec.texp, rec.uv);
// 		if (depth < data.max_depth && rec.mat_ptr->scatterDirection(r.direction, rec, sseed, R, dummy_bool, brdf_scale, v_out))
// 		{
// 			const Ray ref(rec.p, v_out);
// 			c += brdf_scale * R * Color(ref, s, 0.01, FLT_MAX, time, sseed, rseed, depth + 1, 0, false, NULL);
// 		}
// 		return c + data.ambient * rec.mat_ptr->ambientResponse(rec.uvw, r.direction, rec.p, rec.uv);
// 	}
// 	else {
// 		const auto t = 0.5 * (Float3::UnitVector(r.direction).y + 1);
// 		return   (1.0 - t) * make_float3(1.0, 1.0, 1.0) + rec.t * make_float3(50 / 255.0, 130 / 255.0, 255 / 255.0);
// 	}
// }


float3 RTRenderer::PathRenderer::Shade(Ray& r, SurfaceHitRecord& rec, float tmin, float tmax, float time, float2& sseed, float2& rseed, int depth, int spec_depth, bool corl,
float3& factor, const RTSamplerData* data)
{
	bool dummy_bool = true;
	float brdf_scale;		
	float3 R;
	if (rec.t < 99999)
	{
		float3 v_out;
		auto c = make_float3(0, 0, 0);
		c += rec.mat_ptr->emittedRadiance(rec.uvw, -r.direction, rec.texp, rec.uv);
		if (depth < data->max_depth && rec.mat_ptr->scatterDirection(r.direction, rec, sseed, R, dummy_bool, brdf_scale, v_out))
		{
			r = Ray(rec.p, v_out);
			factor *= brdf_scale * R;

			return make_float3(0, 0, 0);
			
			//c += brdf_scale * R * Color(r, s, 0.01, FLT_MAX, time, sseed, rseed, depth + 1, 0, false, NULL);
		}
		//return c + data.ambient * rec.mat_ptr->ambientResponse(rec.uvw, r.direction, rec.p, rec.uv);
	}
	else {
		const auto t = 0.5 * (Float3::UnitVector(r.direction).y + 1);
		return factor*  (1.0 - t) * make_float3(1.0, 1.0, 1.0) + rec.t * make_float3(50 / 255.0, 130 / 255.0, 255 / 255.0);
	}
}
