#include "Scene.h"
#include "BVH.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <iostream>
#include "MyModel.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

HostScene* HostScene::instance = nullptr;

void HostScene::LoadObj(std::string filename, int* mat, int mat_count, float size)
{
	printf(CYN "[CPU]" YEL "Loading Obj Model:%s ...\n" RESET, filename);
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> material;
	std::string err;
	std::string warn;
	//bool ret = tinyobj::LoadObj(&attrib, &shapes, &material, &err, filename.c_str());
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &material, &warn, &err, filename.c_str());


	if (!warn.empty())std::cout << warn ;
	if (!err.empty()) std::cerr << err ;
	if (!ret) exit(1);

	printf("Materials count: %d shapes count: %d \n", material.size(), shapes.size());
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
		for (int i = 0; i < vertice.size(); i += 3)triangles.push_back(new Triangle(*vertice[i + 1], *vertice[i], *vertice[i + 2], s > mat_count ? mat[0] : mat[s]));
	}



	printf(CYN "[CPU]" GRN "Load Obj Model:%s Completed\n" RESET, filename);

}

void HostScene::LoadTextures(char** imageFilenames,int textureCount)
{

	for (auto i = 0; i < textureCount; i++) {
		int width, height, depth;
		const auto tex_data = stbi_load(imageFilenames[i], &width, &height, &depth, 0);
		const auto size = width * height * depth;
		auto h_data = new float[size];
		printf(CYN "[CPU]" YEL "Loading Texture: %s (%d,%d,%d)\n" RESET, imageFilenames[i], width, height, depth);
		for (unsigned int layer = 0; layer < 3; layer++)
			for (auto i = 0; i < static_cast<int>(width * height); i++)h_data[layer * width * height + i] = tex_data[i * 3 + layer] / 255.0;

		//cudaArray Descriptor
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		//cuda Array
		cudaArray* d_cuArr;
		cudaMalloc3DArray(&d_cuArr, &channelDesc, make_cudaExtent(width, height, 3), cudaArrayLayered);


		cudaMemcpy3DParms myparms = { 0 };
		myparms.srcPos = make_cudaPos(0, 0, 0);
		myparms.dstPos = make_cudaPos(0, 0, 0);
		myparms.srcPtr = make_cudaPitchedPtr(h_data, width * sizeof(float), width, height);
		myparms.dstArray = d_cuArr;
		myparms.extent = make_cudaExtent(width, height, 3);
		myparms.kind = cudaMemcpyHostToDevice;
		cudaMemcpy3D(&myparms);


		cudaResourceDesc    texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = d_cuArr;
		cudaTextureDesc     texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		//texDescr.normalizedCoords = false;
		texDescr.filterMode = cudaFilterModeLinear;
		texDescr.addressMode[0] = cudaAddressModeWrap;   // clamp
		texDescr.addressMode[1] = cudaAddressModeWrap;
		texDescr.addressMode[2] = cudaAddressModeWrap;
		texDescr.readMode = cudaReadModeElementType;
		texDescr.normalizedCoords = true;
		cudaCreateTextureObject(&textlist[i], &texRes, &texDescr, nullptr);

	}
	printf(CYN"[CPU]" GRN "LoadTexture Completed\n" RESET);
}

void HostScene::Load(std::string filename)
{ 
	material_count = 6;
	materials = new Material[material_count]
	{

		Material(metal, new float[4]{ 0.7, 0.7, 0.7, 0.5f }),
		Material(metal, new float[4]{ 1, 1, 1, 0.1f }),
		Material(lambertian, new float[3]{ 1, 1, 1 }),
		Material(dielectirc, new float[1]{1.5f }),
		Material(metal, new float[4]{ 1, 1, 1, 0.0f }),
		Material(lambertian, new float[3]{ 1, 0, 0}),
	};

	materials = new Material[material_count]
	{

		Material(lambertian, new float[3]{ 1, 1, 1 }),
		Material(lambertian, new float[3]{ 1, 0, 0 }),
		Material(lambertian, new float[3]{ 0, 0, 1 }),
		Material(light, new float[4]{ 1, 1, 1,25 }),
		Material(dielectirc, new float[1]{1.5f }),
		Material(metal, new float[4]{ 1, 1, 1, 0.0f }),
	};
	auto textureCount = 1;
	char** imageFilenames =new char*[textureCount]
	{
		//"images/BG.jpg",
		"images/black.jpg",
		// "images/BG2.jpg",
	};
	LoadTextures(imageFilenames,textureCount);

	lookat = make_float3(0, 5, 0);
	LoadObj("models/cornell/CornellBox-Original.obj", new int[7] {0,0,0,2,1,0,3},7, 5);

	// lookat = make_float3(3, 8, 0);
	// LoadMesh("bunny", 2, triangles,10);



}

void HostScene::Build()
{
	printf(CYN "[CPU]" YEL "Building BVH...\n" RESET);
	bvh = BuildBVH(triangles.data(), triangles.size());
	Print(bvh,true);
	printf(CYN "[CPU]" GRN "Build BVH Completed\n" RESET);
}
