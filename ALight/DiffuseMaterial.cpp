
#include "DiffuseMaterial.h"
#include "float2Extension.h"
#include "SurfaceHitRecord.h"
float3 DiffuseMaterial::ambientResponse(const ONB&, const float3&,
	const float3& p, const float2& uv)
{
	return R->value(uv, p);
}

bool DiffuseMaterial::explicitBrdf(const ONB&, const float3&,
	const float3&, const float3& p, const float2& uv, float3& brdf)
{
	float k = .318309886184f; // 1.0 / M_PI
	brdf = k * R->value(uv, p);
	return true;
}

bool DiffuseMaterial::scatterDirection(const float3& in_dir,
	const SurfaceHitRecord& rec, float2& seed, float3& color, bool& CEL,
	float& brdf, float3& v_out)
{
	ONB uvw = rec.uvw;
	float3 p = rec.p;
	float2 uv = rec.uv;

	CEL = false;
	brdf = 1.0f;

	const float two_pi = 6.28318530718f;
	const float phi = two_pi * seed.x;
	const float r = sqrt(seed.y);
	float x = r * cos(phi);
	float y = r * sin(phi);
	float z = sqrt(1 - x * x - y * y);

	color = R->value(uv, p);
	v_out = x * uvw.u() + y * uvw.v() + z * uvw.w();

	Float2::Scramble(seed);
	return true;
}

