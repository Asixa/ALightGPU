#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdlib.h"
#include <thread>

#include "header/Hitable.h"
#include "header/Camera.h"
#include "header/GLWindow.h"
#include "curand_kernel.h"
using namespace std;

Camera cam;
GPUHitable *gpu_world;
void InitData()
{
	PixelLength = ImageHeight * ImageWidth * 4;
	PixelData = new GLbyte[PixelLength];
	for (auto i = 0; i < PixelLength; i++)
		PixelData[i] = static_cast<GLbyte>(int(0));
}

__device__ bool HitTest(GPUHitable* list, int size, Ray r, float t_min, float t_max,  HitRecord& rec)
{
	HitRecord temp_rec;
	auto hit_anything = false;
	double closest_so_far = t_max;
	for (auto i=0;i<size;i++)
	{
		if (list[i].Hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

__global__ 
void Sampler(int d_width, int d_height, int worldsize, GPUHitable* d_world, byte * d_pixeldata, Camera* d_camera, curandState *const rngStates)
{
	// Determine thread ID
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto tid2 = blockIdx.y * blockDim.y + threadIdx.y;
	// Initialise the RNG
	const unsigned int seed = 1234;
	//***	curandState random = rngStates[tid];
	curand_init(seed, tid, tid2, &rngStates[tid]);			//初始化随机数
	Vec3 color(0, 0, 0);


	//**********  Debug Specific Pixel **********
	//int x = 256, y = 256;
	//float u = float(x) / float(512);
	//float v = float(y) / float(512);

	const int x = blockIdx.x * 16 + threadIdx.x,y = blockIdx.y * 16 + threadIdx.y;

	for (auto j = 0; j < SPP; j++) {
		const auto u = float(x + curand_uniform(&rngStates[tid])) / float(512),
		v = float(y + curand_uniform(&rngStates[tid])) / float(512);
		Ray ray(Vec3(0, 0, 0), Vec3(-2.0, -2.0, -1.0) + u * Vec3(4, 0, 0) + v * Vec3(0, 4, 0));
		Vec3 c(0, 0, 0);
		float factor = 1;
		for (auto i = 0; i < MAX_SCATTER_TIME; i++)
		{
			HitRecord rec;
			if (HitTest(d_world, worldsize, ray, 0, 99999, rec))
			{
				// random in unit sphere
				Vec3 random_in_unit_sphere;
				do random_in_unit_sphere = 2.0*Vec3(curand_uniform(&rngStates[tid]), curand_uniform(&rngStates[tid]), curand_uniform(&rngStates[tid])) - Vec3(1, 1, 1);
				while (random_in_unit_sphere.squared_length() >= 1.0);

				factor /= 2;
				auto target = rec.p + rec.normal + random_in_unit_sphere;
				ray = Ray(rec.p, target - rec.p);

				//****** 超过最大反射次数，返回黑色 ******
				if (i == MAX_SCATTER_TIME-1)
				{
					c = Vec3(0, 0, 0);
					break;
				}
			}
			else
			{
				const auto t = 0.5*(unit_vector(ray.Direction()).y() + 1);
				c = factor * ((1.0 - t)*Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0));
				break;
			}
		}
		color += c;
	}
	color /= SPP;

	//SetColor
	const auto i = 512 * 4 * y + x * 4;
	d_pixeldata[i] = color.r() * 255;
	d_pixeldata[i + 1] = color.g() * 255;
	d_pixeldata[i + 2] = color.b() * 255;
	d_pixeldata[i + 3] = 255;
}

cudaError_t GPURender()
{
	const auto begin = clock();

	//****** 创建GPU显存指针 ******
	Camera * d_camera;
	int * d_Width = 0;
	int * d_Height = 0;
	byte * d_pixeldata;
	GPUHitable * d_world_gpu;
	const auto h_pixeldata = new byte[ImageWidth*ImageHeight * 4];
	for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)h_pixeldata[i]=byte(0);		//设置像素缓冲初始值

	//***** Init Scene Data ******
	float p1[4] = { 0,0,-1,0.5 };
	float p2[4] = { 0,-100.5,-1,100 };
	GPUHitable w[2] = { GPUHitable(p1),GPUHitable(p2) };
	//****************************

	cout << "准备开始渲染" << endl;
	const auto cuda_status = cudaSetDevice(0);											// Cuda Status for checking error

	curandState *d_rng_states = nullptr; //随机数
	dim3 grid(512 / BlockSize, 512 / BlockSize), block(BlockSize, BlockSize);			// Split area, 32*32/block
	//dim3 grid(1),block(1);  this line for debuging specific pixel

	//******  分配地址 ****** 
	cudaMalloc(reinterpret_cast<void**>(&d_Width), sizeof(int));
	cudaMalloc(reinterpret_cast<void**>(&d_Height), sizeof(int));
	cudaMalloc(reinterpret_cast<void**>(&d_world_gpu), 24*2);
	cudaMalloc(reinterpret_cast<void**>(&d_pixeldata), 512*512*4*sizeof(byte));
	cudaMalloc(reinterpret_cast<void**>(&d_camera), sizeof(cam));
	cudaMalloc(reinterpret_cast<void **>(&d_rng_states), grid.x * block.x * sizeof(curandState));

	//****** 内存复制 host->Device ******
	cudaMemcpy(d_Width, &ImageWidth, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Height, &ImageHeight, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_world_gpu, &w, 24*2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_camera, &cam, sizeof(Camera), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pixeldata, h_pixeldata, 512 * 512 * 4 * sizeof(byte), cudaMemcpyHostToDevice);

	//******分配线程 ******
	Sampler <<<grid,block>>>(512,512,2,d_world_gpu,d_pixeldata,d_camera, d_rng_states);

	//****** 复制内存 Device->host ******
	cudaMemcpy(h_pixeldata, d_pixeldata, 512 * 512 * 4 * sizeof(byte), cudaMemcpyDeviceToHost);

	//****** 转换缓冲数据 ****** TODO 可以优化
	for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)PixelData[i] = h_pixeldata[i];
	
	printf("渲染完成，总消耗时间: %lf秒", double(clock() - begin) / CLOCKS_PER_SEC);
	return cuda_status;
}


int main(int argc, char* argv[])
{
	InitData();
	GPURender();
	InitWindow(argc, argv, GLUT_DOUBLE | GLUT_RGBA, 100, 100, ImageWidth, ImageHeight, "ALightGPU");
	return 0;
}


