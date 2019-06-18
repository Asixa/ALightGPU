#pragma once

enum GPUPtrType
{
	_NULL,
	_BVH,
	_Triangle
};
struct GPUPtr
{
	GPUPtrType type;
	int p;
	void Set(GPUPtrType t,int i)
	{
		type = t; 
		p = i;
	}
	void Set()
	{
		type = _NULL;
		p = 0;
	}
};