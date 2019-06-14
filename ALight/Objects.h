#pragma once

struct Sphere
{
    float3 position;
    float radius;
    float3 albedo;
    float3 specular;
	float smoothness;
	float3 emission;
	__device__ Sphere(float3 p,float r,float3 a,float3 _specular,float _smoothness,float3 _emission):position(p),radius(r),albedo(a),specular(_specular),smoothness(_smoothness),emission(_emission)
	{
		
	}
};

