#include "float2Extension.h"

float Float2::Dot(const float2& v1, const float2& v2)
{
	return v1.x* v2.x + v1.y * v2.y;
}

void Float2::Scramble(float2& v)
{
	auto y = v.x;
	const auto x = v.y * 1234.12345054321f;
	v.x = x - static_cast<int>(x);
	y = y * 7654.54321012345f;
	v.y = y - static_cast<int>(y);
}
