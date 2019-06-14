#include "RTDeviceData.h"

float3 RTDeviceData::SampleTexture(const int index, float u, const float v) const
{
		return  make_float3(
			tex2DLayered<float>(Textures[index], u, 1 - v, 0),
			tex2DLayered<float>(Textures[index], u, 1 - v, 1),
			tex2DLayered<float>(Textures[index], u, 1 - v, 2));
}
