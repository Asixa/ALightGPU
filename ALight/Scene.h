#pragma once
#include "Triangle.h"
#include <vector>
#include "AABB.h"
#include "BVH.h"
#include <string>

class  HostScene
{
	static HostScene* instance;
public:
	std::vector<Triangle*>triangles;
	std::vector<AABB>aabbs;
	BVH* bvh;
	void LoadObj(std::string filename, int mat, float size = 1);
	void Load(std::string filename);
	void Build();
	static HostScene* Instance()
	{
		if (!instance)instance = new HostScene();
		return instance;
	}
};



class  DeviceScene
{
	Triangle* triangles;
};