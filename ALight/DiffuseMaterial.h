#pragma once
#include "Material.h"
#include "Texture.h"

//#include "SimpleTexture.h"

class DiffuseMaterial : public Material
{
public:
	DiffuseMaterial(Texture* t) { R = t; }

	virtual float3 ambientResponse(const ONB&,
		const float3&,                        // incident direction
		const float3&,                        // Texture point
		const float2&);                       // Texture coordinate

	virtual bool explicitBrdf(const ONB&,
		const float3&,                        // unit vector v1
		const float3&,                        // unit vector v0
		const float3&,                        // Texture point
		const float2&,                        // Texture coordinate
		float3&);

	virtual bool scatterDirection(const float3&,// incident Vector
		const SurfaceHitRecord&,               // hit we are shading
		float2&,                              // random seed                    
		float3&,                                  // color to attenuate by
		bool&,                                 // count emitted light?
		float&,                                // brdf scale
		float3&);                             // scattered direction

	Texture* R;
};