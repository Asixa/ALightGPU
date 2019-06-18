#include "RayTracer.h"
#include "DeviceManager.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Setting.h"
#include <direct.h>
#include <string>
#include <algorithm>
#include "Scene.h"
RayTracer::RayTracer() = default;

RayTracer::~RayTracer() = default;

void RayTracer::ReSetIPR()
{
	if (!Setting::IPR)return;
	sampled = 0;
}

RayTracer::RayTracer(const bool GPU):GPU(GPU)
{
	if (GPU)device_manager = new DeviceManager();
}

void RayTracer::Init(GLbyte* d,int w,int h)
{
	width = w;
	height = h;
	data = d;


	//Material
	material_count = 6;
	materials = new Material[material_count]
	{ 
		
		Material(metal, new float[4]{ 0.7, 0.7, 0.7, 0.5f }),
		Material(metal, new float[4]{ 1, 1, 1, 0.1f }),
		Material(lambertian, new float[3]{ 1, 1, 1 }),
		Material(dielectirc, new float[1]{1.5f }),
		Material(metal, new float[4]{ 1, 1, 1, 0.0f }),
		Material(lambertian, new float[3]{ 1, 1, 1}),
	};

	//Textures
	auto textureCount = 1;
	const char* imageFilenames[1] =
	{
		"images/BG5.jpg",
		// "images/BG2.jpg",
	};
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
		cudaArray * d_cuArr;
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



	if (GPU)
	{
		device_manager->Init(this,*HostScene::Instance());
	}
}

void RayTracer::Render() const
{
	if(GPU)
	{
		device_manager->Run();
	}
}

