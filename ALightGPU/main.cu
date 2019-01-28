#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdlib.h"
#include <thread>

#include "Hitable.h"
#include "Camera.h"
#include "GLWindow.h"
#include "curand_kernel.h"
#include "MathHelper.h"
#include "BVH.h"
#include "Sphere.h"
#include <list>
#include <vector>
#include "device.h"
#include "Renderer.h"
#include <helper_cuda.h>
using namespace std;
using namespace Renderer;

void ReSetIPR()
{
	if(!Use_IPR)return;
	current_spp = 0; 
	for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)h_pixeldataF[i] = 0;
}

void Render()
{
	if(!Use_IPR&&current_spp!=0)return;
	//ImageRender();
	IPRRender();
}
void PrintDeviceInfo()
{
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		printf("没有支持CUDA的设备!\n");
		return;
	}

	for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("设备 %d: \"%s\"\n", dev, deviceProp.name);

		char msg[256];
		sprintf_s(msg, sizeof(msg),
			"global memory大小:        %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
		printf("%s", msg);

		printf("SM数:                    %2d \n每SMCUDA核心数:           %3d \n总CUDA核心数:             %d \n",
			deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
			deviceProp.multiProcessorCount);

		printf("静态内存大小:             %zu bytes\n",
			deviceProp.totalConstMem);
		printf("每block共享内存大小:      %zu bytes\n",
			deviceProp.sharedMemPerBlock);
		printf("每block寄存器数:          %d\n",
			deviceProp.regsPerBlock);
		printf("线程束大小:               %d\n",
			deviceProp.warpSize);
		printf("每处理器最大线程数:       %d\n",
			deviceProp.maxThreadsPerMultiProcessor);
		printf("每block最大线程数:        %d\n",
			deviceProp.maxThreadsPerBlock);
		printf("线程块最大维度大小        (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("网格最大维度大小          (%d, %d, %d)\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);

		printf("\n");
	}
	printf("************设备信息打印完毕************\n\n");
}
void OnMouseMove(int x,int y)
{
	cam_rotation[0] += x / 57.3;
	cam_rotation[1] -= y / 57.30f;
	if (cam_rotation[1] > 3.14 / 2)cam_rotation[1] = 3.14 / 2;
	if (cam_rotation[1] < -3.14 / 2)cam_rotation[1] = -3.14 / 2;

	if (cam_rotation[0] > 2 * 3.1415)cam_rotation[0] = 0;
	if (cam_rotation[0] < 0)cam_rotation[0] = 2 * 3.1415;

	camera_lookat = unit_vector(Vec3(cos(cam_rotation[0]), sin(cam_rotation[1]), sin(cam_rotation[0])));
	cam.Update(cam.Origin(), cam.Origin() + camera_lookat, Vec3(0, 1, 0), FOV, float(ImageWidth) / float(ImageHeight));
}

void OnKeyDown()
{
	auto newpos = Vec3(0, 0, 0);
	if (GLWindow::keyDown['w'])newpos += camera_lookat * 0.05f;
	if (GLWindow::keyDown['s'])newpos -= camera_lookat * 0.05f;
	cam.Update(cam.Origin()+ newpos, newpos + cam.Origin() + camera_lookat, Vec3(0, 1, 0), FOV, float(ImageWidth) / float(ImageHeight));
}

int main(int argc, char* argv[])
{
	PrintDeviceInfo();
	InitData();
	Init();
	if(Use_IPR)SPP = IPR_SPP; 
	GLWindow::InitWindow(argc, argv,
		GLUT_DOUBLE | GLUT_RGBA, 100, 100, ImageWidth, ImageHeight, "ALightGPU");
	IPR_Dispose();
	return 0;
}


