#include "Scene.h"
#include "BVH.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <iostream>
#include "MyModel.h"

HostScene* HostScene::instance = nullptr;

void HostScene::LoadObj(std::string filename, int* mat,int mat_count, float size)
{
	printf(CYN "[CPU]" YEL "Loading Obj Model:%s ...\n" RESET, filename);
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> material;
	std::string err;
	std::string warn;
	//bool ret = tinyobj::LoadObj(&attrib, &shapes, &material, &err, filename.c_str());
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &material, &warn, &err, filename.c_str());

	if (!warn.empty()) {
		std::cout << warn << std::endl;
		if (!err.empty()) std::cerr << err << std::endl;
		if (!ret) exit(1);

		printf("Materials count: %d\n shapes count: %d", material.size(),shapes.size());
		auto vertice = std::vector<Vertex*>();

		for (size_t s = 0; s < shapes.size(); s++) {
			vertice.clear();
			size_t index_offset = 0;
			for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
				int fv = shapes[s].mesh.num_face_vertices[f];

				// Loop over vertices in the face.

				for (size_t v = 0; v < fv; v++) {
					tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
					auto vertex = new Vertex();
					vertex->point.x = attrib.vertices[3 * idx.vertex_index + 0] * size;
					vertex->point.y = attrib.vertices[3 * idx.vertex_index + 1] * size;
					vertex->point.z = attrib.vertices[3 * idx.vertex_index + 2] * size;
					if (attrib.normals.size() > 3 * idx.normal_index + 2) {
						vertex->normal.x = attrib.normals[3 * idx.normal_index + 0];
						vertex->normal.y = attrib.normals[3 * idx.normal_index + 1];
						vertex->normal.z = attrib.normals[3 * idx.normal_index + 2];
						
						//printf("normals: %f,%f,%f", vertex->normal.x, vertex->normal.y, vertex->normal.z);
					}
					else
					{
						vertex->normal.x = 0;
						vertex->normal.y = 0;
						vertex->normal.z = 0;
					}
					if (attrib.texcoords.size() > 2 * idx.texcoord_index + 1) {
						vertex->uv.x = attrib.texcoords[2 * idx.texcoord_index + 0];
						vertex->uv.y = attrib.texcoords[2 * idx.texcoord_index + 1];
					}
					else
					{
						vertex->uv.x = 0;
						vertex->uv.y = 0;
					}
					vertice.push_back(vertex);
				}

				index_offset += fv;
			
				// per-face material
				// shapes[s].mesh.material_ids[f];
			}
			for (int i = 0; i < vertice.size(); i += 3)triangles.push_back(new Triangle(*vertice[i + 1], *vertice[i], *vertice[i + 2],s>mat_count?mat[0]:mat[s]));
		}


		
		printf(CYN "[CPU]" GRN "Load Obj Model:%s Completed\n" RESET, filename);
	}
}
void HostScene::Load(std::string filename)
{ 
	lookat = make_float3(3, 8, 0);
	// LoadObj("models/buddha.obj", new int[1] {0},1, 5);
	LoadMesh("bunny", 3, triangles,10);


	// triangles.push_back(
	// 	new Triangle(
	// 		Vertice(make_float3(-1, 0, -5), make_float3(1, 1, 1),make_float2(0,0)),
	// 		Vertice(make_float3(1, 0, -5), make_float3(1, 1, 1),make_float2(0,0)),
	// 		Vertice(make_float3(0, 1 * sqrt(static_cast<float>(2)), -5), make_float3(1, 1, 1),make_float2(0,0))
	// 	));
	//
	//
	// triangles.push_back(
	// 	new Triangle(
	// 		Vertice(make_float3(-1, 0, -6), make_float3(1, 1, 1), make_float2(0, 0)),
	// 		Vertice(make_float3(1, 0, -6), make_float3(1, 1, 1), make_float2(0, 0)),
	// 		Vertice(make_float3(0, 1 * sqrt(static_cast<float>(2)), -6), make_float3(1, 1, 1), make_float2(0, 0))
	// 	));
	//
	//
	// triangles.push_back(
	// 	new Triangle(
	// 		Vertice(make_float3(-1, 0, -7), make_float3(1, 1, 1), make_float2(0, 0)),
	// 		Vertice(make_float3(1, 0, -7), make_float3(1, 1, 1), make_float2(0, 0)),
	// 		Vertice(make_float3(0, 1 * sqrt(static_cast<float>(2)), -7), make_float3(1, 1, 1), make_float2(0, 0))
	// 	));
	// triangles.push_back(
	// 	new Triangle(
	// 		Vertice(make_float3(-1, 0, -8), make_float3(1, 1, 1), make_float2(0, 0)),
	// 		Vertice(make_float3(1, 0, -8), make_float3(1, 1, 1), make_float2(0, 0)),
	// 		Vertice(make_float3(0, 1 * sqrt(static_cast<float>(2)), -8), make_float3(1, 1, 1), make_float2(0, 0))
	// 	));



}

void HostScene::Build()
{
	printf(CYN "[CPU]" YEL "Building BVH...\n" RESET);
	bvh = BuildBVH(triangles.data(), triangles.size());
	Print(bvh);
	printf(CYN "[CPU]" GRN "Build BVH Completed\n" RESET);
}
