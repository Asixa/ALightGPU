#include "DeviceManager.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdlib>
#include <helper_cuda.h>
#include <curand_kernel.h>

#include "RayTracer.h"
#include "Setting.h"
#include "RTSampler.h"
#include <iostream>
#include "Float2Byte.h"
#include "Engine.h"


DeviceManager::DeviceManager()
{
}


DeviceManager::~DeviceManager()
{
}

void DeviceManager::PrintDeviceInfo()
{
	auto device_count = 0;
	cudaGetDeviceCount(&device_count);

	if (device_count == 0)
	{
		printf("û��֧��CUDA���豸!\n");
		return;
	}
	for (auto dev = 0; dev < device_count; dev++)
	{
		cudaSetDevice(dev);
		cudaDeviceProp device_prop{};
		cudaGetDeviceProperties(&device_prop, dev);
		printf("�豸 %d: \"%s\"\n", dev, device_prop.name);
		char msg[256];
		sprintf_s(msg, sizeof(msg),
			"global memory��С:        %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(device_prop.totalGlobalMem / 1048576.0f),
			static_cast<unsigned long long>(device_prop.totalGlobalMem));
		printf("%s", msg);
		printf("SM��:                    %2d \nÿSM CUDA������:           %3d \n��CUDA������:             %d \n",
			device_prop.multiProcessorCount,
			_ConvertSMVer2Cores(device_prop.major, device_prop.minor),
			_ConvertSMVer2Cores(device_prop.major, device_prop.minor) *
			device_prop.multiProcessorCount);
		printf("��̬�ڴ��С:             %zu bytes\n",
			device_prop.totalConstMem);
		printf("ÿblock�����ڴ��С:      %zu bytes\n",
			device_prop.sharedMemPerBlock);
		printf("ÿblock�Ĵ�����:          %d\n",
			device_prop.regsPerBlock);
		printf("�߳�����С:               %d\n",
			device_prop.warpSize);
		printf("ÿ����������߳���:       %d\n",
			device_prop.maxThreadsPerMultiProcessor);
		printf("ÿblock����߳���:        %d\n",
			device_prop.maxThreadsPerBlock);
		printf("�߳̿����ά�ȴ�С        (%d, %d, %d)\n",
			device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1],
			device_prop.maxThreadsDim[2]);
		printf("�������ά�ȴ�С          (%d, %d, %d)\n",
			device_prop.maxGridSize[0], device_prop.maxGridSize[1],
			device_prop.maxGridSize[2]);
		printf("\n");
	}
	printf("************�豸��Ϣ��ӡ���************\n\n");
}

void DeviceManager::Init(RayTracer* tracer)
{
	ray_tracer = tracer;
	grid = dim3(ray_tracer->Width / Setting::BlockSize, ray_tracer->Height / Setting::BlockSize);
	block = dim3(Setting::BlockSize, Setting::BlockSize);

	host_float_data = new float[ray_tracer->Width * ray_tracer->Height * 4];
	cudaMalloc(reinterpret_cast<void**>(&devicde_float_data), ray_tracer->Width * ray_tracer->Height * 4 * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&devicde_byte_data), ray_tracer->Width * ray_tracer->Height * 4 * sizeof(GLbyte));
	cudaMalloc(reinterpret_cast<void**>(&rng_states), grid.x * block.x * sizeof(curandState));
	cudaMalloc(reinterpret_cast<void**>(&d_camera), sizeof(Camera));
}

void DeviceManager::Run()
{
	// printf("Hello");

	//****** �����ڴ� host->device ******
	cudaMemcpy(devicde_float_data, host_float_data, ray_tracer->Width * ray_tracer->Height * 4 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_camera, Engine::Instance()->camera, sizeof(Camera), cudaMemcpyHostToDevice);
	SetConstants();
	//dim3 grid(ray_tracer->Width / Setting::BlockSize, ray_tracer->Height / Setting::BlockSize), block(Setting::BlockSize, Setting::BlockSize);


	ray_tracer->Sampled += Setting::SPP;
	IPRSampler << <grid, block >> > (ray_tracer->Width, ray_tracer->Height, (rand() / (RAND_MAX + 1.0)) * 1000, Setting::SPP, ray_tracer->Sampled, 4, 0, devicde_float_data, rng_states, d_camera);
	Float2Byte <<<grid, block >> > (ray_tracer->Width, ray_tracer->Sampled, Setting::SPP, devicde_float_data, devicde_byte_data);
	
	cudaDeviceSynchronize();
	
	//****** �����ڴ� Device->host ******
	cudaMemcpy(host_float_data, devicde_float_data, ray_tracer->Width * ray_tracer->Height * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(ray_tracer->Data, devicde_byte_data, ray_tracer->Width * ray_tracer->Height * 4 * sizeof(GLbyte), cudaMemcpyDeviceToHost);

	const auto error = cudaGetLastError();
	if (error != 0)printf("Cuda Error %d\n", error);
}
