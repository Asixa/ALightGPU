#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdlib.h"
#include <thread>

#include "header/Hitable.h"
#include "header/Camera.h"
#include "header/Sphere.h"
#include "header/hitable_list.h"
#include "header/GLWindow.h"
#include "curand_kernel.h"
using namespace std;

Hitable *world;	Hitable *list[2]; Camera cam;
GPUHitable *gpu_world;
void InitData()
{
	PixelLength = ImageHeight * ImageWidth * 4;
	PixelData = new GLbyte[PixelLength];
//	col = new Vec3[ImageHeight*ImageWidth];
	//InitWindow zero
	for (auto i = 0; i < PixelLength; i++)
		PixelData[i] = static_cast<GLbyte>(int(0));
}
void InitScene()
{
	// list[0] = new Sphere(Vec3(0, 0, -1), 0.5);
	// list[1] = new Sphere(Vec3(0, -100.5, -1), 100);
	// world = new HitableList(list, 2);



	//gpu_world = w;
}



__device__ Vec3 shadeNormal(const Ray& r, Hitable *world)
{
	// HitRecord rec;
	// if (world->Hit(r, 0, 99999, rec))
	// {
	// 	return  0.5*Vec3(rec.normal.x() + 1, rec.normal.y() + 1, rec.normal.z() + 1);
	// }
	// else
	// {
	// 	Vec3 unit_dir = unit_vector(r.Direction());
	// 	auto t = 0.5*(unit_dir.y() + 1);
	// 	return  (1.0 - t)*Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
	// }
}

__device__ bool HitTest(GPUHitable* list, int size, Ray r, float t_min, float t_max,  HitRecord& rec)
{
	HitRecord temp_rec;
	auto hit_anything = false;
	double closest_so_far = t_max;
	for (auto i=0;i<size;i++)
	{
		//printf("Check Hit %d\n", i);
		if (list[i].Hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

// __device__ Vec3 shade(const Ray& r, GPUHitable *world,int worldSize,int Depth)
// {
// 	if (Depth == 0)return Vec3(0, 0, 0);
// 	Depth--;
//
// 	if(Depth==0)
// 	{
// 		
// 		printf("第二次");
// 		return Vec3(0, 1, 0);
//
// 	}
// 	//printf("Start Shade\n");
// 	HitRecord rec;
// 	//printf("Initized rec\n");
// 	bool hit = HitTest(world, worldSize, r, 0, 99999, rec);
// 	//printf(hit?" hit\n":"Nothit\n");
// 	if(!hit)
// 	{
// 		//printf("如果没有hit\n");
// 		Vec3 unit_dir = unit_vector(r.Direction());
// 		//printf("归一化完成\n");
// 		auto t = 0.5*(unit_dir.y() + 1);
// 		return  (1.0 - t)*Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
// 		//printf("设置参数完成\n");
// 	}
// 	else
// 	{
// 		Vec3 r = RandomInUnitSphere();
// 		
// 	
// 		Vec3 target = rec.p + rec.normal + r;
// 		// printf("random {%f,%f,%f}\n", r.x(), r.y(), r.z());
// 		// printf("point {%f,%f,%f}\n", rec.p.x(), rec.p.y(), rec.p.z());
// 		// printf("normal {%f,%f,%f}\n", rec.normal.x(), rec.normal.y(), rec.normal.z());
// 		// printf("target {%f,%f,%f}\n", target.x(), target.y(), target.z());
// 		//return Vec3(1, 0, 0);
// 		auto ray = Ray(rec.p, target - rec.p);
//
// 		printf("ray {%f,%f,%f}{%f,%f,%f}\n", ray.A.x(), ray.A.y(), ray.A.z(), ray.B.x(), ray.B.y(), ray.B.z());
// 		
// 		auto next = shade(ray, world, worldSize,  Depth);
// 		printf("next {%f,%f,%f}\n", next.x(), next.y(), next.z());
// 		return 0.5f*next;
// 	}
// }
//
// __device__ Vec3 shadeNoRecursion(const Ray& r, GPUHitable *world, int worldSize, int Depth)
// {
// 	Ray ray = r;
// 	Vec3 color(0, 0, 0);
// 	float factor = 1;
// 	for (int i=0;i<8;i++)
// 	{
// 		printf("ray {%f,%f,%f}{%f,%f,%f}\n", ray.A.x(), ray.A.y(), ray.A.z(), ray.B.x(), ray.B.y(), ray.B.z());
// 		HitRecord rec;
// 		if(HitTest(world, worldSize, ray, 0, 99999, rec))
// 		{
// 			auto t = 0.5*(unit_vector(r.Direction()).y() + 1);
// 			color+=factor*((1.0 - t)*Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0));
// 		}
// 		else
// 		{
// 			factor /= 2;
// 			Vec3 target = rec.p + rec.normal + RandomInUnitSphere();;
// 			ray = Ray(rec.p, target - rec.p);		
// 		}
// 	}
//
//
//
// 	if (Depth == 0)return Vec3(0, 0, 0);
// 	Depth--;
//
// 	if (Depth == 0)
// 	{
//
// 		printf("第二次");
// 		return Vec3(0, 1, 0);
//
// 	}
// 	//printf("Start Shade\n");
// 	HitRecord rec;
// 	//printf("Initized rec\n");
// 	bool hit = HitTest(world, worldSize, r, 0, 99999, rec);
// 	//printf(hit?" hit\n":"Nothit\n");
// 	if (!hit)
// 	{
// 		//printf("如果没有hit\n");
// 		Vec3 unit_dir = unit_vector(r.Direction());
// 		//printf("归一化完成\n");
// 		auto t = 0.5*(unit_dir.y() + 1);
// 		return  (1.0 - t)*Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
// 		//printf("设置参数完成\n");
// 	}
// 	else
// 	{
// 		Vec3 r = RandomInUnitSphere();
//
//
// 		Vec3 target = rec.p + rec.normal + r;
// 		// printf("random {%f,%f,%f}\n", r.x(), r.y(), r.z());
// 		// printf("point {%f,%f,%f}\n", rec.p.x(), rec.p.y(), rec.p.z());
// 		// printf("normal {%f,%f,%f}\n", rec.normal.x(), rec.normal.y(), rec.normal.z());
// 		// printf("target {%f,%f,%f}\n", target.x(), target.y(), target.z());
// 		//return Vec3(1, 0, 0);
// 		auto ray = Ray(rec.p, target - rec.p);
//
// 		printf("ray {%f,%f,%f}{%f,%f,%f}\n", ray.A.x(), ray.A.y(), ray.A.z(), ray.B.x(), ray.B.y(), ray.B.z());
//
// 		auto next = shade(ray, world, worldSize, Depth);
// 		printf("next {%f,%f,%f}\n", next.x(), next.y(), next.z());
// 		return 0.5f*next;
// 	}
// }


__global__ 
void Sampler(int d_width, int d_height, int worldsize, GPUHitable* d_world, byte * d_pixeldata, Camera* d_camera, curandState *const rngStates)
{
	//printf("复合球体数据%f,%f,%f,%f\n", d_world[0].data[0], d_world[0].data[1], d_world[0].data[2], d_world[0].data[3]);
	// Determine thread ID
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid2 = blockIdx.y * blockDim.y + threadIdx.y;
	// Initialise the RNG
	unsigned int seed = 1234;
	curandState random = rngStates[tid];
	curand_init(seed, tid, tid2, &rngStates[tid]);
	Vec3 c(0, 0, 0);

	//int x = 256, y = 256;
	//float u = float(x) / float(512);
	//float v = float(y) / float(512);

	 int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;

	//printf("开始调用Shade %f\n",ray.A.x());

	//printf("开始调用Shade %f\n", ray.A.x());

	//着色
	//c += shadeNoRecursion(ray, d_world, worldsize,2);

	// for (auto i=0;i<10;i++)
	// {
	// 	printf("random: %f\n", curand_uniform(&rngStates[tid]));
	// }


	unsigned long long _seed = 4;

	for (int j = 0; j < SPP; j++) {

		float u = float(x + curand_uniform(&rngStates[tid])) / float(512);
		float v = float(y + curand_uniform(&rngStates[tid])) / float(512);
		Ray ray(Vec3(0, 0, 0), Vec3(-2.0, -2.0, -1.0) + u * Vec3(4, 0, 0) + v * Vec3(0, 4, 0));
		Vec3 _c(0, 0, 0);
		float factor = 1;
		for (int i = 0; i <= 8; i++)
		{
			//printf("ray {%f,%f,%f}{%f,%f,%f}\n", ray.A.x(), ray.A.y(), ray.A.z(), ray.B.x(), ray.B.y(), ray.B.z());
			HitRecord rec;
			if (HitTest(d_world, worldsize, ray, 0, 99999, rec))
			{
				Vec3 random_in_unit_sphere;
				do random_in_unit_sphere = 2.0*Vec3(curand_uniform(&rngStates[tid]), curand_uniform(&rngStates[tid]), curand_uniform(&rngStates[tid])) - Vec3(1, 1, 1);
				while (random_in_unit_sphere.squared_length() >= 1.0);

				factor /= 2;
				Vec3 target = rec.p + rec.normal + random_in_unit_sphere;
				ray = Ray(rec.p, target - rec.p);
				if (i == 7)
				{
					_c = Vec3(0, 0, 0);
					break;
				}
			}
			else
			{
				auto t = 0.5*(unit_vector(ray.Direction()).y() + 1);
				_c = factor * ((1.0 - t)*Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0));
				break;
			}
		}
		
		c += _c;
	}
	c /= SPP;













	//printf("UV:%f,%f,Color:%f,%f,%f\n", u, v, c.x(), c.y(), c.z());

	//c = Vec3(float(x) / float(ImageWidth), float(y) / float(512), 0.2f);

	//SetColor
	const auto i = 512 * 4 * y + x * 4;
	//Changes[width * y + x]++;
	d_pixeldata[i] = c.r() * 255;
	d_pixeldata[i + 1] = c.g() * 255;
	d_pixeldata[i + 2] = c.b() * 255;
	d_pixeldata[i + 3] = 255;
}

cudaError_t GPURender()
{
	//Hitable * d_world;
	Camera * d_camera;
	int * d_Width = 0;
	int * d_Height = 0;
	cudaError_t cudaStatus;
	byte * d_pixeldata;
	GPUHitable * d_world_gpu;

	byte* h_pixeldata = new byte[ImageWidth*ImageHeight * 4];
	for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)h_pixeldata[i]=byte(0);

	//***** scene
	float p1[4] = { 0,0,-1,0.5 };
	float p2[4] = { 0,-100.5,-1,100 };
	GPUHitable w[2] = { GPUHitable(p1),GPUHitable(p2) };
	//****

	cout << sizeof(gpu_world) << "  " << sizeof(*gpu_world) << sizeof(GPUHitable) << "  " << endl;
	cout << "准备" << endl;
	cudaStatus = cudaSetDevice(0);

	
	curandState *d_rngStates = 0; //随机数
	dim3 grid(512 / BlockSize, 512 / BlockSize), block(BlockSize, BlockSize);
	//dim3 grid(1),block(1);

	//分配地址
	cudaMalloc(reinterpret_cast<void**>(&d_Width), sizeof(int));
	cudaMalloc(reinterpret_cast<void**>(&d_Height), sizeof(int));
	//cudaMalloc(reinterpret_cast<void**>(&d_world), sizeof(world));
	cudaMalloc(reinterpret_cast<void**>(&d_world_gpu), 24*2);
	cudaMalloc(reinterpret_cast<void**>(&d_pixeldata), 512*512*4*sizeof(byte));
	cudaMalloc(reinterpret_cast<void**>(&d_camera), sizeof(cam));
	cudaMalloc(reinterpret_cast<void **>(&d_rngStates), grid.x * block.x * sizeof(curandState));

	

	//内存复制 host->Device
	cudaMemcpy(d_Width, &ImageWidth, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Height, &ImageHeight, sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_world, world, sizeof(world), cudaMemcpyHostToDevice);
	cudaMemcpy(d_world_gpu, &w, 24*2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_camera, &cam, sizeof(Camera), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pixeldata, h_pixeldata, 512 * 512 * 4 * sizeof(byte), cudaMemcpyHostToDevice);
	//分配线程
	

	Sampler <<<grid,block>>>(512,512,2,d_world_gpu,d_pixeldata,d_camera, d_rngStates);
	//复制内存 Device->host
	cudaMemcpy(h_pixeldata, d_pixeldata, 512 * 512 * 4 * sizeof(byte), cudaMemcpyDeviceToHost);
    //for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)cout << int(h_pixeldata[i])<<",";

	for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)PixelData[i] = h_pixeldata[i];
	cout << "完成" << endl;
	
	return cudaStatus;
}


int main(int argc, char* argv[])
{
	InitData();
	InitScene();
	clock_t begin = clock();
	GPURender();
	printf("总消耗时间: %lf", double(clock() - begin) / CLOCKS_PER_SEC);
	InitWindow(argc, argv, GLUT_DOUBLE | GLUT_RGBA, 100, 100, ImageWidth, ImageHeight, "ALightGPU");
	return 0;
}


