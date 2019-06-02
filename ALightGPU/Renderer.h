#pragma once
#include "root.h"
#include <cuda_runtime_api.h>
#include "Material.h"
#include "Sphere.h"
#include "BVH.h"
#include "device.h"

#define STB_IMAGE_IMPLEMENTATION

#include <iostream>
#include <vector>
#include "stb_image.h"
#include "Model.h"
#include "MathHelper.h"

namespace Renderer
{
	float FOV = 75;
	Vec3 cam_rotation(0, 0, 0), camera_lookat(0, 0, 0);
	Camera cam(Vec3(-2, 1, 1), Vec3(0, 0, -1), Vec3(0, 1, 0),FOV, float(ImageWidth) / float(ImageHeight));
	float * h_pixeldataF;
	bool Use_IPR = true;


	curandState *d_rng_states = nullptr;


	const int material_count = 5;

	float m1[MATERIAL_PARAMTER_COUNT] = { 3,0,0,0.5f,0,0 };
	float m2[MATERIAL_PARAMTER_COUNT] = { 2,0,0.6,0.5,0,0 };
	float m3[MATERIAL_PARAMTER_COUNT] = { 1,1,1,0.1,0,0 };
	float m4[MATERIAL_PARAMTER_COUNT] = { 1,1,0,0,0,0 };
	float m5[MATERIAL_PARAMTER_COUNT] = { 0,0.1,0.1,0.5,0,0 };
	Material m[material_count] = { 
		Material(dielectirc,m1),
		Material(lambertian,m2) ,
		Material(metal,m3) ,
		Material(lambertian,m4),
		Material(lambertian,m5) };

	const char *imageFilenames[] =
	{
		"D:/Codes/Projects/Academic/ComputerGraphic/ALightGPU/bin/win64/Release/earthmap.png",
		"D:/Codes/Projects/Academic/ComputerGraphic/ALightGPU/bin/win64/Release/marsmap.jpg",
		"D:/Codes/Projects/Academic/ComputerGraphic/ALightGPU/bin/win64/Release/moonmap.jpg"
	};

	float * d_pixeldata;

	Material * d_materials;
	Camera * d_camera;
	#if !RenderDEBUG 
	dim3 grid(ImageWidth / BlockSize, ImageHeight / BlockSize), block(BlockSize, BlockSize);
	//dim3 grid(1, 1), block(BlockSize, BlockSize);
	#endif
	#if  RenderDEBUG 
	dim3 grid(1, 1), block(1, 1);
	#endif
	Hitable ** d_objs;
	Hitable ** d_new_world;

	int bvh_index = 0;
	int BVH_Root_ID = -1;

	Hitable** ObjList;
	Hitable** FinalList;
	int object_count =0;
	BVHNode h_BVHRoot;
	cudaTextureObject_t textlist[TEXTURE_COUNT];
	std::vector<Hitable*>hitables;
	DeviceData d_data;

	void Scene1()
	{
		
		//unsigned char *tex_data = stbi_load("earthmap.jpg", &nx, &ny, &nn, 0);
		object_count = 5;
		ObjList = new Hitable*[object_count];
		ObjList[0] = new Sphere(Vec3(0, 0, 0.5), 1.079/8, 1);//moon
		//ObjList[1] = new Sphere(Vec3(0, -100.5, -1), 100, 1);
		ObjList[1] = new Sphere(Vec3(0, 0, 0), 2.106/8, 3);//mars
		ObjList[2] = new Sphere(Vec3(0, 0, -1), 3.959/8, 4);//earth
		ObjList[3] = new Sphere(Vec3(0, -100.5, 0), 100, 2);
		ObjList[4] = new Sphere(Vec3(-1, 0, -2.106 / 16), 0.5, 0);
		// ObjList[5] = new Triangle(
		// 	Vertex(Vec3(-1, 1, 0), Vec3(0, 1, 0), 0, 0),
		// 	Vertex(Vec3(1, 1, 0), Vec3(0, 1, 0), 0, 1),
		// 	Vertex(Vec3(0, 1, 2), Vec3(0, 1, 0),1,1),4);

		h_BVHRoot = BVHNode(ObjList, object_count, 0, 1);
	}
	void Scene0()
	{

		//unsigned char *tex_data = stbi_load("earthmap.jpg", &nx, &ny, &nn, 0);
		object_count = 4;
		ObjList = new Hitable*[object_count];
		ObjList[0] = new Sphere(Vec3(0, -(0.5 - 1.079 / 8), 0.5), 1.079 / 8, 1);//moon
		//ObjList[1] = new Sphere(Vec3(0, -100.5, -1), 100, 1);
		ObjList[1] = new Sphere(Vec3(0, -(0.5- 2.106 / 8), 0), 2.106 / 8, 3);//mars
		ObjList[2] = new Sphere(Vec3(0, -(0.5 - 3.959 / 8), -1), 3.959 / 8, 4);//earth
		ObjList[3] = new Sphere(Vec3(0, -100.5, 0), 100, 2);
		// ObjList[5] = new Triangle(
		// 	Vertex(Vec3(-1, 1, 0), Vec3(0, 1, 0), 0, 0),
		// 	Vertex(Vec3(1, 1, 0), Vec3(0, 1, 0), 0, 1),
		// 	Vertex(Vec3(0, 1, 2), Vec3(0, 1, 0),1,1),4);

		h_BVHRoot = BVHNode(ObjList, object_count, 0, 1);
	}
	void Scene2()
	{
		object_count = 485;
		int n = 500;
		ObjList = new Hitable*[n + 1];
		ObjList[0] = new Sphere(Vec3(0, -1000, 0), 1000, 1);
		//new lambertian(Vec3(0.5, 0.5, 0.5)));
		int i = 1;
		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				float choose_mat = drand48();
				Vec3 center(a + 0.9*drand48(), 0.2, b + 0.9*drand48());
				if ((center - Vec3(4, 0.2, 0)).Length() > 0.9) {
					if (choose_mat < 0.8) {  // diffuse
						ObjList[i++] = new Sphere(center, 0.2, 3);
						// new lambertian(Vec3(drand48()*drand48(), drand48()*drand48(), drand48()*drand48())));
					}
					else if (choose_mat < 0.95) { // metal
						ObjList[i++] = new Sphere(center, 0.2,
							//new metal(Vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())), 0.5*drand48()));
							2);
					}
					else {  // glass
						ObjList[i++] = new Sphere(center, 0.2, 0);
						//new dielectric(1.5));
					}
				}
			}
		}

		ObjList[i++] = new Sphere(Vec3(0, 1, 0), 1.0, 0);
		//new dielectric(1.5));
		ObjList[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, 1);
		//new lambertian(Vec3(0.4, 0.2, 0.1)));
		ObjList[i++] = new Sphere(Vec3(4, 1, 0), 1.0, 2);
		//new metal(Vec3(0.7, 0.6, 0.5), 0.0));

		printf("count %d", i);

		h_BVHRoot = BVHNode(ObjList, object_count, 0, 1);
	}
	void Scene3()
	{
	
		h_BVHRoot = LoadMesh("bunny_lowpoly", 4,hitables,ObjList);
		object_count = hitables.size();
		//object_count = 20000;
		printf("[%d]", object_count);
		//h_BVHRoot = BVHNode(ObjList, object_count, 0, 1);
	}

	inline void InitTextureList()
	{
		for (auto i = 0; i < TEXTURE_COUNT; i++) {
			int width, height, depth;
			const auto tex_data = stbi_load(imageFilenames[i],&width, &height, &depth, 0);
			const auto size = width * height * depth;
			float* h_data = new float[size];
			printf("LoadTexture %d,%d,%d\n", width, height, depth);
			for (unsigned int layer = 0; layer < 3; layer++)
				for (auto i = 0; i < static_cast<int>(width * height); i++)h_data[layer*width*height + i] = tex_data[i * 3 + layer] / 255.0;

			//cudaArray Descriptor
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
			//cuda Array
			cudaArray *d_cuArr;
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
			cudaCreateTextureObject(&textlist[i], &texRes, &texDescr, NULL);
		}
	}
	void InitData()
	{
		const size_t newHeapSize = 4608ull * 1024ull * 1024ull;;
		cudaDeviceSetLimit(cudaLimitStackSize, newHeapSize);
		printf("Adjusted heap size to be %d\n", newHeapSize);

		h_pixeldataF = new float[ImageWidth*ImageHeight * 4];
		for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)h_pixeldataF[i] = 0;

		PixelLength = ImageHeight * ImageWidth * 4;
		PixelData = new GLbyte[PixelLength];
		for (auto i = 0; i < PixelLength; i++)
			PixelData[i] = static_cast<GLbyte>(int(0));
		Scene3();
	}

	inline void AddBvh(BVHNode* bvh)
	{
		FinalList[(bvh_index++) + object_count] = bvh;
		//TODO 以下未经测试
		if (bvh->Left->type == Instance::BVH)AddBvh(static_cast<BVHNode*>(bvh->Left));
		if (bvh->Right->type == Instance::BVH)AddBvh(static_cast<BVHNode*>(bvh->Right));
	}

	inline void SetupBVH()
	{
		const auto bvh_count = h_BVHRoot.count();
		const auto total = bvh_count + object_count;
		FinalList = new Hitable*[total];
		//for (auto i = 0; i < object_count; i++)FinalList[i] = ObjList[i];
		for (auto i = 0; i < object_count; i++)FinalList[i] = hitables[i];

		AddBvh(&h_BVHRoot);
	

		for (auto i = 0; i < total; i++)FinalList[i]->id = i;
		for (auto i = 0; i < total; i++)FinalList[i]->SetChildId();

		BVH_Root_ID = h_BVHRoot.id;

		cudaMalloc(reinterpret_cast<void **>(&d_objs), total * sizeof(Hitable*));
		cudaMalloc(reinterpret_cast<void **>(&d_new_world), total * sizeof(Hitable*));
		//cudaMemcpy(d_objs, FinalList, total * sizeof(Hitable*), cudaMemcpyHostToDevice);

		//Set Object Array
		for (int i = 0; i < total; i++)
		{
			Hitable *tmp;
			if (FinalList[i]->type == 1) {
				cudaMalloc(reinterpret_cast<void **>(&tmp), sizeof(Sphere));
				cudaMemcpy(tmp, FinalList[i], sizeof(Sphere), cudaMemcpyHostToDevice);
			}
			else if (FinalList[i]->type == 2) {
				cudaMalloc(reinterpret_cast<void **>(&tmp), sizeof(BVHNode));
				cudaMemcpy(tmp, FinalList[i], sizeof(BVHNode), cudaMemcpyHostToDevice);
			}
			else if (FinalList[i]->type == 3) {
				cudaMalloc(reinterpret_cast<void **>(&tmp), sizeof(Triangle));
				cudaMemcpy(tmp, FinalList[i], sizeof(Triangle), cudaMemcpyHostToDevice);
			}
			ArraySetter << <1, 1 >> > (d_objs, i, tmp);
		}

		//Fix
		//cudaMalloc(reinterpret_cast<void**>(&d_bvh_root), sizeof(BVHNode));
		//cudaMemcpy(d_bvh_root, &h_BVHRoot, sizeof(BVHNode), cudaMemcpyHostToDevice);

		printf("Total %d\n", total);
		auto block = total / 1024+1;
		auto thread = total / block+1;
		printf("devide %d,%d\n", block,thread);
		WorldArrayFixer << <block, thread >> > (d_objs, d_new_world,total);
		auto error = cudaGetLastError();
		if (error != 0)
			printf("error at WorldArrayFixer %d\n", error);

	}
	
	void Init()
	{
		SetupBVH();
		InitTextureList();
		//******  分配地址 ****** 
		cudaMalloc(reinterpret_cast<void**>(&d_materials), sizeof(Material)*material_count);
		cudaMalloc(reinterpret_cast<void**>(&d_pixeldata), ImageWidth * ImageHeight * 4 * sizeof(float));
		cudaMalloc(reinterpret_cast<void**>(&d_camera), sizeof(Camera));
		if (current_spp == 0) {
			if (d_rng_states != nullptr)cudaFree(d_rng_states);
			cudaMalloc(reinterpret_cast<void **>(&d_rng_states), grid.x * block.x * sizeof(curandState));
		}

		//****** 内存复制 host->Device ******);
		cudaMemcpy(d_materials, &m, sizeof(Material) *material_count, cudaMemcpyHostToDevice);

		d_data.world = d_new_world;
		d_data.materials = d_materials;

		for (int i = 0; i < TEXTURE_COUNT; i++) {
			d_data.texs[i] = textlist[i];
		}

		printf("初始化显存完成\n");

	}
	void IPR_Dispose()
	{
		cudaDeviceReset();
		//****** 释放显存 **********
		cudaFree(d_pixeldata);
		cudaFree(d_materials);
		cudaFree(d_camera);

	}

	int z = 0;
	cudaError_t ImageRender()
	{
		const auto cuda_status = cudaSetDevice(0);
		if (z == 0)z = 1;
		else { return  cuda_status; }
	
		// InitTexture();
		//TextureLab << <grid, block >> > (d_pixeldata, ImageWidth, ImageHeight, 0);
		cudaMemcpy(h_pixeldataF, d_pixeldata, ImageWidth * ImageHeight * 4 * sizeof(float), cudaMemcpyDeviceToHost);

		for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)
			PixelData[i] = h_pixeldataF[i]  * 255;
		return cuda_status;
	}
	cudaError_t IPRRender()
	{
		const int random_seed = drand48() * 1000;
		const auto cuda_status = cudaSetDevice(0);

		//更新数据

		cudaMemcpy(d_camera, &cam, sizeof(Camera), cudaMemcpyHostToDevice);
		cudaMemcpy(d_pixeldata, h_pixeldataF, ImageWidth * ImageHeight * 4 * sizeof(float), cudaMemcpyHostToDevice);
		//******分配线程 ******


		IPRSampler << <grid, block >> > (
			ImageWidth, ImageHeight,
			random_seed,
			SPP,
			MAX_SCATTER_TIME,
			BVH_Root_ID,
			d_pixeldata,
			d_camera,
			d_rng_states,
			d_data);

		//****** 复制内存 Device->host ******
		cudaMemcpy(h_pixeldataF, d_pixeldata, ImageWidth * ImageHeight * 4 * sizeof(float), cudaMemcpyDeviceToHost);

		//****** 转换缓冲数据 ****** TODO 可以优化
		for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)
			PixelData[i] = (Range(sqrt(h_pixeldataF[i] / (current_spp == 0 ? SPP : current_spp + SPP))) * 255);
		current_spp += SPP;
		auto error = cudaGetLastError();
		//cudaError_t;
		if(error!=0)
		printf("error %d\n", error);
		return cuda_status;
	}
}
