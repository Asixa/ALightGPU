#pragma once
#include "Hitable.h"
#include "Vertex.h"

#define EPSILON  1e-4f
class Triangle:public Hitable
{
public:
	Vertex v0, v1, v2;
	Vec3 GNormal;
	int mat_id;

	Triangle(Vertex a, Vertex b, Vertex c, int mat):v0(a),v1(b),v2(c),mat_id(mat)
	{
		GNormal = (a.normal + b.normal + c.normal) / 3;
	}


};
