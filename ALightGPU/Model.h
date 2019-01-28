#pragma once
#include "Vertex.h"
#include <fstream>

#include <string>
#include "BVH.h"
#include "Triangle.h"
#include <iostream>
#include "Renderer.h"
const std::string ModelPath = "Models/";
const std::string ModelExtension = ".GPUModel";
typedef unsigned char CSbyte;
inline Vertex ReadVertex(std::ifstream * read_stream)
{
	Vertex vertex;
	read_stream->read(reinterpret_cast<char*>(&vertex.point[0]), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.point[1]), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.point[2]), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.normal[0]), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.normal[1]), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.normal[2]), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.uv.X), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.uv.Y), sizeof(float));
	return vertex;
}

inline BVHNode LoadMesh(const std::string name,int mat,std::vector<Hitable*>& list,Hitable** tris)
{
	//std::ifstream read_stream(ModelPath + name + ModelExtension, std::ios::binary);
	std::ifstream read_stream("D:/" + name + ModelExtension, std::ios::binary);
	if (!read_stream.good())return nullptr;
	CSbyte m;
	read_stream >> m;
	if (m != CSbyte(233))
	{
		printf("模型魔数不正确! %d",m);
		return nullptr;
	}
	int ver_count, mesh_count,matid;
	read_stream.read(reinterpret_cast<char *>(&mesh_count), sizeof(int));
	read_stream.read(reinterpret_cast<char *>(&matid), sizeof(int));
	read_stream.read(reinterpret_cast<char *>(&ver_count), sizeof(int));
	printf("mesh %d  mat %d  vertice %d", mesh_count,matid,ver_count);
	//const auto triangles = new Hitable*[ver_count / 3];
	tris = new Hitable*[ver_count / 3];
	for (auto i = 0; i < ver_count; i+=3)
	{
		//printf("Load Triangle %d",i/3);
		tris[i / 3] = new Triangle(
			ReadVertex(&read_stream),
			ReadVertex(&read_stream),
			ReadVertex(&read_stream), mat);
		list.push_back(tris[i / 3]);
	}

	// for(int i=0;i<ver_count/3;i++)
	// {
	// 	printf("%d,", tris[i]->type);
	// }

	//return BVHNode(tris, 20000, 0.001f, 99999);
	return BVHNode(tris,ver_count/3,0.001f,99999);
}