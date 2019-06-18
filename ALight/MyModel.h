#pragma once
#pragma once

#include <fstream>

#include <string>
#include "BVH.h"
#include "Triangle.h"
#include <iostream>
#include "Renderer.h"
#include <vector>
#include "Defines.h"
const std::string ModelPath = "Models/";
const std::string ModelExtension = ".GPUModel";
typedef unsigned char CSbyte;
inline Vertice ReadVertex(std::ifstream* read_stream,float size)
{
	Vertice vertex;
	read_stream->read(reinterpret_cast<char*>(&vertex.point.x), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.point.y), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.point.z), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.normal.x), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.normal.y), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.normal.z), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.uv.x), sizeof(float));
	read_stream->read(reinterpret_cast<char*>(&vertex.uv.y), sizeof(float));
	vertex.point *= size;
	return vertex;
}

inline void LoadMesh(const std::string name, int mat, std::vector<Triangle*>& list, float  size=1)
{
	std::ifstream read_stream("D:/" + name + ModelExtension, std::ios::binary);
	if (!read_stream.good())return;
	CSbyte m;
	read_stream >> m;
	if (m != CSbyte(233))
	{
		printf(RED "[ERROR]Wrong Magic Number %d" RESET, m);
		return;
	}
	int ver_count, mesh_count, matid;
	read_stream.read(reinterpret_cast<char*>(&mesh_count), sizeof(int));
	read_stream.read(reinterpret_cast<char*>(&matid), sizeof(int));
	read_stream.read(reinterpret_cast<char*>(&ver_count), sizeof(int));
	printf(CYN"[CPU]" YEL "Loading Model: %s... (Mesh %d mat %d vertice %d)\n" RESET, name, mesh_count, matid, ver_count);
	for (auto i = 0; i < ver_count; i += 3)
	{
		list.push_back(new Triangle(
			ReadVertex(&read_stream,size),
			ReadVertex(&read_stream, size),
			ReadVertex(&read_stream, size), mat));
	}
	printf(CYN"[CPU]" GRN "Load Model %s Completed\n" RESET, name);
}