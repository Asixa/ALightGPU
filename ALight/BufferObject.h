#pragma once
#include <stddef.h>

enum BufferObjectType
{
	Sphere,
	Triangle
};

struct BufferObject
{
public:
	BufferObjectType type=Sphere;
	float data[];

};

