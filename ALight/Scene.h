#pragma once
#include "Triangle.h"
#include <vector>
#include "AABB.h"
#include "BVH.h"
#include <string>
#include "Material.h"

class  HostScene
{
	
public:
	static HostScene* instance;
	std::vector<Triangle*>triangles;
	std::vector<AABB>aabbs;
	BVH* bvh;
	float3 lookat;
	Material* materials;
	int material_count;
	cudaTextureObject_t textlist[1];
	bool ground = false;
	void LoadTextures(char** imageFilenames, int textureCount);
	void LoadObj(std::string filename, int* mat, int mat_count, float size = 1);
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