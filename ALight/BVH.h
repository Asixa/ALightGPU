#pragma once
#include "AABB.h"
#include "Triangle.h"
struct Ray;
struct RTDeviceData;
struct  BVH;

int box_x_compare(const void* a, const void* b);
int box_y_compare(const void* a, const void* b);
int box_z_compare(const void* a, const void* b);
BVH* BuildBVH(Triangle* tri);
BVH* BuildBVH(Triangle** list, int n);
BVH* ToDevice(BVH*);
__host__ __device__ void Print(BVH* bvh,bool root=false);
struct Ray;
struct  BVH
{
	bool tri;
	Triangle* triangle; 
	AABB* aabb;	
	BVH* left;		
	BVH* right;			
	BVH(){}
};
