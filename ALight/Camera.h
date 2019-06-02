#pragma once
#include <driver_functions.h>
#include "Defines.h"
#include "float3Extension.h"

struct Camera
{
	float3 Origin,LowerLeftCorner,Horizontal,Vertical;
	Camera(const float3 lookfrom, const float3 lookat, const float3 vup, const float fov, const float aspect)
	{
		const float theta = fov * M_PI / 180;
		const float half_height = tan(theta / 2);
		const auto half_width = aspect * half_height;
		Origin = lookfrom;
		const auto w = Float3::UnitVector(lookfrom - lookat);
		const auto u = Float3::UnitVector(Float3::Cross(vup, w));
		const auto v = Float3::Cross(w, u);
		LowerLeftCorner = make_float3(-half_width, -half_height, -1.0);
		LowerLeftCorner = Origin - half_width * u - half_height * v - w;
		Horizontal = 2 * half_width * u;
		Vertical = 2 * half_height * v;
	}
};
