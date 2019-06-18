#include "Scene.h"
#include "BVH.h"
#include "tiny_obj_loader.h"
#include <iostream>
#include "MyModel.h"

HostScene* HostScene::instance = nullptr;


void HostScene::Load(std::string filename)
{

	LoadMesh("bunny", 5, triangles,10);
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



	// std::vector<tinyobj::shape_t> shapes;
	// std::vector<tinyobj::material_t> material;
	// std::string err;
	// bool ret = tinyobj::LoadObj(&mAttributes, &shapes, &material, &err, filename.c_str());
	//
	// if (!err.empty()) { // `err` may contain warning message.
	// 	std::cerr << err << std::endl;
	// }
	// if (!ret) {
	// 	exit(1);
	// }
	//
	// for (size_t s = 0; s < shapes.size(); s++) {
	//
	// 	size_t index_offset = 0;
	// 	for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
	// 		auto triangle=new Triangle();
	// 		// TriangleIndices index;
	// 		// index.a = shapes[s].mesh.indices[index_offset + 0];
	// 		// index.b = shapes[s].mesh.indices[index_offset + 1];
	// 		// index.c = shapes[s].mesh.indices[index_offset + 2];
	//
	// 		triangle->v1= shapes[s].mesh.num_face_vertices.
	//
	// 		triangle->v1.point = shapes[s].mesh.indices[index_offset + 0].vertex_index;
	// 		triangles.push_back(triangle);
	// 		//t_indices.push_back(index);
	// 		//material_ids.push_back(shapes[s].mesh.material_ids[f]);
	// 		index_offset += 3;
	// 	}
	// }
}

void HostScene::Build()
{
	printf(CYN "[CPU]" YEL "Building BVH...\n" RESET);
	bvh = BuildBVH(triangles.data(), triangles.size());
	printf(CYN "[CPU]" GRN "Build BVH Completed\n" RESET);
}
