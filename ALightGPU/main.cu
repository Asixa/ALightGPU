#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdlib.h"
#include <thread>

#include "header/Hitable.h"
#include "header/Camera.h"
#include "header/GLWindow.h"
#include "curand_kernel.h"
using namespace std;


GPUHitable *gpu_world;
void InitData()
{
	PixelLength = ImageHeight * ImageWidth * 4;
	PixelData = new GLbyte[PixelLength];
	for (auto i = 0; i < PixelLength; i++)
		PixelData[i] = static_cast<GLbyte>(int(0));
}

__device__ bool HitTest(GPUHitable* list, int size, Ray r, float t_min, float t_max,  HitRecord& rec, Material* materials)
{
	HitRecord temp_rec;
	auto hit_anything = false;
	double closest_so_far = t_max;
	for (auto i=0;i<size;i++)
	{
		if (list[i].Hit(r, t_min, closest_so_far, temp_rec, materials)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

__global__ 
void Sampler(int d_width, int d_height, int worldsize, GPUHitable* d_world, byte * d_pixeldata, Camera* d_camera, curandState *const rngStates, Material* materials)
{
	// Determine thread ID
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto tid2 = blockIdx.y * blockDim.y + threadIdx.y;
	// Initialise the RNG
	const unsigned int seed = 1234;
	//***	curandState random = rngStates[tid];
	curand_init(seed, tid, tid2, &rngStates[tid]);			//初始化随机数
	Vec3 color(0, 0, 0);


	

	const int x = blockIdx.x * 16 + threadIdx.x,y = blockIdx.y * 16 + threadIdx.y;
	//**********  Debug Specific Pixel **********
	//int x = 256, y = 256;
	//float u = float(x) / float(512);
	//float v = float(y) / float(512);
	for (auto j = 0; j < SPP; j++) {
		const auto u = float(x + curand_uniform(&rngStates[tid])) / float(512),
		v = float(y + curand_uniform(&rngStates[tid])) / float(512);
		
		//Ray ray(Vec3(0, 0, 0), Vec3(-2.0, -2.0, -1.0) + u * Vec3(4, 0, 0) + v * Vec3(0, 4, 0));

		// printf("origin  %f,%f,%f LowerLeftCorner  %f,%f,%f  Horizontal()  %f,%f,%f Vertical %f,%f,%f\n", 
		// 	d_camera->Origin().x(), d_camera->Origin().y(), d_camera->Origin().z(),
		// 	d_camera->LowerLeftCorner().x(), d_camera->LowerLeftCorner().y(), d_camera->LowerLeftCorner().z(),
		// 	d_camera->Horizontal().x(), d_camera->Horizontal().y(), d_camera->Horizontal().z(),
		// 	d_camera->Vertical().x(), d_camera->Vertical().y(), d_camera->Vertical().z());

		Ray ray(d_camera->Origin(), d_camera->LowerLeftCorner() + u * d_camera->Horizontal() + v * d_camera->Vertical()-d_camera->Origin());
		//Ray ray = d_camera->GetRay(u, v);
		Vec3 c(0, 0, 0);
		Vec3 factor(1,1,1);
		for (auto i = 0; i < MAX_SCATTER_TIME; i++)
		{
			HitRecord rec;
			if (HitTest(d_world, worldsize, ray, 0, 99999, rec, materials))
			{
				// random in unit sphere
				Vec3 random_in_unit_sphere;
				do random_in_unit_sphere = 2.0*Vec3(curand_uniform(&rngStates[tid]), curand_uniform(&rngStates[tid]), curand_uniform(&rngStates[tid])) - Vec3(1, 1, 1);
				while (random_in_unit_sphere.squared_length() >= 1.0);

				//factor /= 2;
				Ray scattered;
				Vec3 attenuation;
				if (i < MAX_SCATTER_TIME&&rec.mat_ptr->scatter(ray,rec,attenuation,scattered,random_in_unit_sphere)) {
					//auto target = rec.p + rec.normal + random_in_unit_sphere;
					factor *= attenuation;
					ray = scattered;
				}
				//****** 超过最大反射次数，返回黑色 ******
				else
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
	
	int * d_Width = 0;
	int * d_Height = 0;
	byte * d_pixeldata;
	GPUHitable * d_world_gpu;
	Material * d_materials;
	Camera * d_camera;
	const auto h_pixeldata = new byte[ImageWidth*ImageHeight * 4];
	for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)h_pixeldata[i]=byte(0);		//设置像素缓冲初始值

	//***** Init Scene Data ******
	Camera cam(Vec3(-2, 1, 1), Vec3(0, 0, -1), Vec3(0, 1, 0), 90, float(ImageWidth) / float(ImageHeight));


	const int object_count = 4;
	float p1[HITABLE_PARAMTER_COUNT] = { 0,0,-1,0.5,0 };
	float p2[HITABLE_PARAMTER_COUNT] = { 0,-100.5,-1,100,1};
	float p3[HITABLE_PARAMTER_COUNT] = { -1,0,-1,0.5,2 };
	float p4[HITABLE_PARAMTER_COUNT] = { 1,0,-1,0.5,3};
	GPUHitable w[object_count] = { GPUHitable(p1),GPUHitable(p2),GPUHitable(p3),GPUHitable(p4) };
	
	const int material_count = 4;
	float m1[MATERIAL_PARAMTER_COUNT] = { 1.5f,0,0,0.5f,0,0};
	float m2[MATERIAL_PARAMTER_COUNT] = { 1,1,1,0.5,0,0};
	float m3[MATERIAL_PARAMTER_COUNT] = { 0,0,1,0.1,0,0 };
	float m4[MATERIAL_PARAMTER_COUNT] = { 0,1,0,0,0,0 };
	Material m[material_count] = { Material(DIELECTIRC,m1),Material(METAL,m2) ,Material(METAL,m3) ,Material(LAMBERTIAN,m4)};
	//****************************

	cout << "准备开始渲染" << endl;
	const auto cuda_status = cudaSetDevice(0);											// Cuda Status for checking error

	curandState *d_rng_states = nullptr; //随机数
	dim3 grid(512 / BlockSize, 512 / BlockSize), block(BlockSize, BlockSize);			// Split area, 32*32/block
	//dim3 grid(1),block(1);  //this line for debuging specific pixel

	//******  分配地址 ****** 
	cudaMalloc(reinterpret_cast<void**>(&d_Width), sizeof(int));
	cudaMalloc(reinterpret_cast<void**>(&d_Height), sizeof(int));
	cudaMalloc(reinterpret_cast<void**>(&d_world_gpu), sizeof(GPUHitable) * object_count);
	cudaMalloc(reinterpret_cast<void**>(&d_materials), sizeof(Material)*material_count);
	cudaMalloc(reinterpret_cast<void**>(&d_pixeldata), 512*512*4*sizeof(byte));
	cudaMalloc(reinterpret_cast<void**>(&d_camera), sizeof(Camera));
	cudaMalloc(reinterpret_cast<void **>(&d_rng_states), grid.x * block.x * sizeof(curandState));

	//****** 内存复制 host->Device ******
	cudaMemcpy(d_Width, &ImageWidth, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Height, &ImageHeight, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_world_gpu, &w, sizeof(GPUHitable) * object_count, cudaMemcpyHostToDevice);
	cudaMemcpy(d_materials, &m, sizeof(Material) *material_count, cudaMemcpyHostToDevice);
	cudaMemcpy(d_camera, &cam, sizeof(Camera), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pixeldata, h_pixeldata, 512 * 512 * 4 * sizeof(byte), cudaMemcpyHostToDevice);

	//******分配线程 ******
	Sampler <<<grid,block>>>(512,512, object_count,d_world_gpu,d_pixeldata,d_camera, d_rng_states,d_materials);

	//****** 复制内存 Device->host ******
	cudaMemcpy(PixelData, d_pixeldata, 512 * 512 * 4 * sizeof(byte), cudaMemcpyDeviceToHost);

	//****** 转换缓冲数据 ****** TODO 可以优化
	//for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)PixelData[i] = h_pixeldata[i];

	//****** 释放显存 **********
	cudaFree(d_Width);
	cudaFree(d_Height);
	cudaFree(d_pixeldata);
	cudaFree(d_world_gpu);
	cudaFree(d_materials);
	cudaFree(d_camera);

	printf("渲染完成，总消耗时间: %lf秒", double(clock() - begin) / CLOCKS_PER_SEC);
	return cuda_status;
}


int main(int argc, char* argv[])
{
	InitData(); 
	thread t(GPURender);
	InitWindow(argc, argv, GLUT_DOUBLE | GLUT_RGBA, 100, 100, ImageWidth, ImageHeight, "ALightGPU");
	return 0;
}


