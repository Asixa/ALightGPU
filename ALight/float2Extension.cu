#include "float2Extension.h"

float Float2::Dot(const float2& v1, const float2& v2)
{
	return v1.x* v2.x + v1.y * v2.y;
}
