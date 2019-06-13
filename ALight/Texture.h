#pragma once
#include "float3Extension.h"

class Texture {
public:

	virtual float3 value(const float2&, const float3&) const = 0;

};
