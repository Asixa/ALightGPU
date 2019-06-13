#pragma once
#include "float3Extension.h"
// #include "SurfaceHitRecord.h"
#include "ONB.h"
class SurfaceHitRecord;
class Material
{
public:
	__device__ virtual bool emits() const { return false; }

	__device__ virtual float3 emittedRadiance(const ONB&,      // ONB of hit point
		const float3&,								// outgoing direction from light
		const float3&,								// Texture point 
		const float2&)								// Texture coordinate 
	{
		return make_float3(0, 0, 0);
	}

	__device__ virtual float3 ambientResponse(const ONB&,      // ONB of hit point
		const float3&,								// incident direction
		const float3&,								// Texture point
		const float2&)								// Texture coordinate
	{
		return make_float3(0, 0, 0);
	}

	__device__ virtual bool explicitBrdf(const ONB&,			// ONB of hit point
		const float3&,								// outgoing vector v0
		const float3&,								// outgoing vector v1
		const float3&,								// Texture point
		const float2&,								// Texture coordinate
		float3&) {
		return false;
	}

	__device__ virtual bool scatterDirection(const float3&,	// incident Vector
		const SurfaceHitRecord&,					// hit we are shading
		float2&,									// random seed                    
		float3&,									// color to attenuate by
		bool&,										// count emitted light?
		float&,										// brdf scale 
		float3&) = 0;								// scattered direction

	__device__ virtual bool isSpecular() { return false; }
	__device__ virtual bool isTransmissive() { return false; }
	__device__ virtual int causticPhotons() { return 0; }
	__device__ virtual int globalPhotons() { return 0; }
	__device__ virtual float3 photonColor() { return make_float3(0.0f, 0.0f, 0.0f); }
};
